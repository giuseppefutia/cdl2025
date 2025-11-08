from __future__ import annotations

import asyncio
import logging
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Tuple

from langchain_openai import ChatOpenAI
from neo4j.exceptions import ClientError as Neo4jClientError

from llm.tool import build_ontology_mapper_tool
from util.config_loader import load_config_api
from util.api_client import ApiClient

# Configure logging to suppress overly verbose output from libraries
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

def ontology_mapper_factory(base_importer_cls, backend: str, config_path: str = "config.ini"):

    async def _abatch_llm(tool, payloads, max_concurrency: int = 8):
        return await tool.abatch(payloads, config={"max_concurrency": max_concurrency})

    class EmbedAPI:
        def __init__(self, api):
            self.api = api

        def embed(self, text: str) -> List[float]:
            # returns a single embedding vector
            r = self.api.post('/embed', {'input': [text]})
            return r[0]['data'][0]

        def embed_many(self, texts: List[str]) -> List[Sequence[float]]:
            if not texts:
                return []
            resp = self.api.post('/embed', {'input': texts})
            vectors = resp[0]["data"]
            if len(vectors) != len(texts):
                raise RuntimeError(f"Embedding count mismatch ({len(vectors)} != {len(texts)})")
            return vectors

    @dataclass
    class Candidate:
        id: str
        label: str
        score: float

    @dataclass
    class MappingDecision:
        best_id: Optional[str]
        best_label: Optional[str]
        confidence: float
        rationale: str
        support: dict[str, Any]

    class OntologyMapper(base_importer_cls):

        def __init__(self):
            super().__init__()
            self.backend = backend
            self.batch_size = 32
            self.cfg_emb = load_config_api("embedding", path=config_path)
            self.emb_api = EmbedAPI(ApiClient(self.cfg_emb))

            # LLM Model Initialization
            url_llm = load_config_api("llm", path=config_path)
            self.llm = ChatOpenAI(
                api_key="EMPTY",
                base_url=url_llm,
                model_name="google/medgemma-4b-it",
                temperature=0,
                max_tokens=8192,
                top_p=0.9,
                stop=["<end_of_turn>", "</s>", "\nUser:", "\n\nUser:"],
                frequency_penalty=0.2,
                presence_penalty=0.0,
            )
            self.ontology_mapper_tool = build_ontology_mapper_tool(self.llm)

            # Configuration parameters
            self.source_label: str = "IcdDisease"
            self.target_label: str = "HpoPhenotype"
            self.target_index_name: str = "hpo_phenotype_embedding"
            self.relationship_type: str = "ICD_MAPS_TO_HPO_PHENOTYPE"
            self.relationship_conf_prop: str = "confidence"
            self.llm_threshold: float = 0.7
            self.k: int = 5
            self.write_relationships: bool = True
            self.mapper_task: str = "icd_to_hpo_phenotype"

        def md_to_param(self, md: MappingDecision) -> Dict[str, Any]:
            """ Convert MappingDecision dataclass to dict for Neo4j parameters. """
            d = asdict(md)  # dataclass -> plain dict
            d["support"] = json.dumps(d.get("support", {}), ensure_ascii=False)  # map -> JSON string
            return {k: v for k, v in d.items() if v is not None}

        ##############################################
        ### Candidate Selection (single and batch) ###
        ##############################################

        def select_candidates(self, text) -> List[Candidate]:
            embedding = self.emb_api.embed(text)
            query = """
            CALL db.index.vector.queryNodes(
                $index, $k, $qe
            ) YIELD node, score
            RETURN node.id AS id, node.label AS label, score
            ORDER BY score DESC
            LIMIT $k
            """
            with self._driver.session(database=self._database) as session:
                result = session.run(query,
                                     index=self.target_index_name,
                                     k=self.k,
                                     qe=embedding)
                candidates = []
                for record in result:
                    candidates.append(
                        Candidate(
                            id=record["id"],
                            label=record["label"],
                            score=record["score"]
                        )
                    )
                return candidates

        def select_candidates_in_batch(self, labels: List[str]) -> Dict[str, List[Candidate]]:
            """
            labels: list of source concept labels. Example: ["Syncope and collapse", "Headache"]
            returns: dictionary {label -> [Candidate, ...]}.
            """
            if not labels:
                return {}

            embeddings = self.emb_api.embed_many(labels)
            items = [{"key": labels[i], "qe": embeddings[i]} for i in range(len(labels))]

            query = """
            UNWIND $items AS row
            CALL (row) {
            CALL db.index.vector.queryNodes($index, $k, row.qe)
                YIELD node, score
            RETURN collect({id: node.id, label: node.label, score: score}) AS topk
            }
            RETURN row.key AS key, topk[0..$k] AS topk
            """

            out: Dict[str, List[Candidate]] = {}
            with self._driver.session(database=self._database) as session:
                result = session.run(query, items=items, index=self.target_index_name, k=self.k)
                for record in result:
                    cands = [Candidate(**c) for c in record["topk"]]
                    out[record["key"]] = cands
            return out

        ############################################
        ### Retrieve context: (single and batch) ###
        ############################################

        def build_context(self, source_id: str, candidates: List[Candidate]) -> Dict[str, Any]:
            if self.mapper_task == "icd_to_hpo_phenotype":
                source_context_query = """
                MATCH (d:IcdDisease {id: $id})
                OPTIONAL MATCH (g:IcdGroup)-[:GROUP_HAS_DISEASE]->(d)
                OPTIONAL MATCH (c:IcdChapter)-[:CHAPTER_HAS_DISEASE]->(d)
                WITH d,
                     collect(DISTINCT g.groupName)   AS gnames,
                     collect(DISTINCT c.chapterName) AS cnames
                RETURN {
                  id: d.id,
                  name: d.label,
                  parentName: d.parentLabel,
                  group:   { groupName: head(gnames) },
                  chapter: { chapterName: head(cnames) }
                } AS context;
                """
                with self._driver.session(database=self._database) as session:
                    result = session.run(source_context_query, id=source_id)
                    record = result.single()
                    source_context = record["context"] if record else {}

                candidate_contexts = []
                for cand in candidates:
                    candidate_context_query = """
                    MATCH (p:HpoPhenotype {id: $id})
                    RETURN {
                      id: p.id,
                      label: p.label,
                      exactSynonym: p.hasExactSynonym,
                      description: p.comment,
                      comment: p.iAO_0000115
                    } AS context;
                    """
                    with self._driver.session(database=self._database) as session:
                        result = session.run(candidate_context_query, id=cand.id)
                        record = result.single()
                        cand_context = record["context"] if record else {}
                        candidate_contexts.append(cand_context)

                return {
                    "source": source_context,
                    "candidates": candidate_contexts
                }

            # Default empty
            return {"source": {}, "candidates": []}

        def build_context_in_batch(
            self,
            sources: List[Tuple[str, str]],  # list of (source_id, source_label)
            cand_map: Dict[str, List[Candidate]]  # from select_candidates_in_batch
        ) -> Dict[str, Dict[str, Any]]:
            """
            Returns:
            {
              source_id: {
                "source": {...},
                "candidates": [ {...}, ... ]
              },
              ...
            }
            """
            if not sources:
                return {}

            source_ids = [sid for sid, _ in sources]

            # 1) fetch source contexts in one query
            q_source = """
            UNWIND $ids AS id
            MATCH (d:IcdDisease {id: id})
            OPTIONAL MATCH (g:IcdGroup)-[:GROUP_HAS_DISEASE]->(d)
            OPTIONAL MATCH (c:IcdChapter)-[:CHAPTER_HAS_DISEASE]->(d)
            WITH d, collect(DISTINCT g.groupName) AS gnames, collect(DISTINCT c.chapterName) AS cnames
            RETURN d.id AS id, {
              id: d.id,
              name: d.label,
              parentName: d.parentLabel,
              group:   { groupName: head(gnames) },
              chapter: { chapterName: head(cnames) }
            } AS context;
            """
            src_ctx: Dict[str, Dict[str, Any]] = {}
            with self._driver.session(database=self._database) as session:
                res = session.run(q_source, ids=source_ids)
                for r in res:
                    src_ctx[r["id"]] = r["context"]

            # 2) gather ALL unique candidate ids across batch
            all_cand_ids: List[str] = []
            for _, label in sources:
                for c in cand_map.get(label, []):
                    all_cand_ids.append(c.id)
            uniq_cand_ids = list(dict.fromkeys(all_cand_ids))

            cand_ctx_map: Dict[str, Dict[str, Any]] = {}
            if uniq_cand_ids:
                q_cand = """
                UNWIND $ids AS id
                MATCH (p:HpoPhenotype {id: id})
                RETURN p.id AS id, {
                  id: p.id,
                  label: p.label,
                  exactSynonym: p.hasExactSynonym,
                  description: p.comment,
                  comment: p.iAO_0000115
                } AS context
                """
                with self._driver.session(database=self._database) as session:
                    res = session.run(q_cand, ids=uniq_cand_ids)
                    for r in res:
                        cand_ctx_map[r["id"]] = r["context"]

            # 3) stitch per-source
            out: Dict[str, Dict[str, Any]] = {}
            for sid, label in sources:
                cands = cand_map.get(label, [])
                out[sid] = {
                    "source": src_ctx.get(sid, {}),
                    "candidates": [cand_ctx_map.get(c.id, {}) for c in cands],
                }
            return out

        ##################################################
        ### Disambiguation context: (single and batch) ###
        ##################################################

        def disambiguate_candidates(
            self,
            source_concept: str,
            source_context: Dict[str, Any],
            candidates: List[Dict[str, Any]]
        ) -> MappingDecision:
            payload = {
                "source_concept": source_concept, 
                "source_context": json.dumps(source_context, ensure_ascii=False),
                "candidate_list": json.dumps(candidates, ensure_ascii=False),
            }
            response = self.ontology_mapper_tool.invoke(payload)
            try:
                return MappingDecision(
                    best_id=response.get("best_id"),
                    best_label=response.get("best_label"),
                    confidence=float(response.get("confidence", 0.0)),
                    rationale=response.get("rationale", ""),
                    support=response.get("support", {}) or {},
                )
            except Exception as e:
                logging.error(f"Failed to parse LLM response: {e}")
                return MappingDecision(
                    best_id=None,
                    best_label=None,
                    confidence=0.0,
                    rationale="Failed to parse LLM response.",
                    support={}
                )

        def disambiguate_candidates_batch_sync(
            self,
            items: Iterable[Tuple[str, dict, list]],  # (source_concept, source_context, candidate_contexts)
            max_concurrency: int = 8,
        ) -> List[MappingDecision]:
            """
            Batch version. Returns one MappingDecision per item, same order.
            """
            payloads = []
            for source_concept, source_context, candidates in items:
                payloads.append({
                    "source_concept": source_concept,
                    "source_context": json.dumps(source_context, ensure_ascii=False),
                    "candidate_list": json.dumps(candidates, ensure_ascii=False),
                })

            # run async abatch from sync code
            raw_responses = asyncio.run(_abatch_llm(self.ontology_mapper_tool, payloads, max_concurrency))

            out: List[MappingDecision] = []
            for resp in raw_responses:
                try:
                    out.append(MappingDecision(
                        best_id=resp.get("best_id"),
                        best_label=resp.get("best_label"),
                        confidence=float(resp.get("confidence", 0.0)),
                        rationale=resp.get("rationale", ""),
                        support=resp.get("support", {}) or {},
                    ))
                except Exception as e:
                    logging.error(f"Failed to parse LLM response: {e}")
                    out.append(MappingDecision(
                        best_id=None,
                        best_label=None,
                        confidence=0.0,
                        rationale="Failed to parse LLM response.",
                        support={}
                    ))
            return out

        ##############################################
        ### End-to-end disambiguation (single/batch) ###
        ##############################################

        def run_disambiguation(self, source_id: str, source_label: str) -> MappingDecision:
            candidates = self.select_candidates(source_label)
            context = self.build_context(source_id=source_id, candidates=candidates)
            source_context = context.get("source", {})
            candidate_contexts = context.get("candidates", [])
            res_disambiguation = self.disambiguate_candidates(
                source_concept=source_label,
                source_context=source_context,
                candidates=candidate_contexts
            )
            logging.debug(f"Disambiguation result: {res_disambiguation}")
            return res_disambiguation

        def run_disambiguation_in_batch(
            self,
            sources: List[Tuple[str, str]],  # [(source_id, source_label), ...]
            max_concurrency: int = 8
        ) -> List[Tuple[str, str, MappingDecision]]:
            """
            Returns a list of (source_id, source_label, MappingDecision) in the same order as `sources`.
            """
            if not sources:
                return []

            # 1) Select candidates in batch using labels
            labels = [lbl for _, lbl in sources]
            cand_map = self.select_candidates_in_batch(labels)

            # 2) Build contexts in batch (source + candidate contexts)
            ctx_map = self.build_context_in_batch(sources, cand_map)  # {source_id: {"source": {...}, "candidates": [...] }}

            # 3) Prepare payloads for LLM disambiguation
            items = []
            for sid, lbl in sources:
                sc = ctx_map.get(sid, {}).get("source", {})
                cc = ctx_map.get(sid, {}).get("candidates", [])
                items.append((lbl, sc, cc))

            # 4) Batch disambiguate (sync wrapper over async abatch)
            decisions = self.disambiguate_candidates_batch_sync(items, max_concurrency=max_concurrency)

            # 5) Return aligned triplets
            out: List[Tuple[str, str, MappingDecision]] = []
            for (sid, lbl), md in zip(sources, decisions):
                out.append((sid, lbl, md))
            
            return out

        ##############################################
        ### Orchestration: read, filter, and write ###
        ##############################################

        def test(self):
            # Single example (optional)
            source = ("R55", "Syncope and collapse")
            res = self.run_disambiguation(source_id=source[0], source_label=source[1])
            print(f"### Disambiguation (single) result for '{source[1]}': {res.best_id} ({res.best_label}), conf={res.confidence}")

            # Batch example (recommended)
            sources = [("R55", "Syncope and collapse"), ("G44.1", "Vascular headache")]
            results = self.run_disambiguation_in_batch(sources)
            for sid, lbl, md in results:
                print(f"### Disambiguation (batch) result for '{lbl}': {md.best_id} ({md.best_label}), conf={md.confidence}")

        def count_missing_source_nodes(self) -> int:
            query = f"""
            MATCH (n:{self.source_label})
            WHERE NOT "ProcessedWithOntologyMapper" IN labels(n)
            RETURN count(n) AS cnt
            """
            with self._driver.session(database=self._database) as session:
                result = session.run(query)
                record = result.single()
                return record["cnt"] if record else 0

        def get_source_nodes(self) -> Generator[Dict[str, str]]:
            query = f"""
            MATCH (n:{self.source_label})
            WHERE NOT n:ProcessedWithOntologyMapper
            RETURN n.id AS id, coalesce(n.label, n.name) AS label
            """
            with self._driver.session(database=self._database) as session:
                result = session.run(query)
                for record in result:
                    print(record)
                    md = self.run_disambiguation(
                        source_id=record["id"],
                        source_label=record["label"]
                    )
                    if md.confidence >= self.llm_threshold:
                        yield {
                            "id": record["id"],
                            "label": record["label"],
                            "disambiguation_result": self.md_to_param(md)
                        }
        
        def get_source_nodes_in_batch(self, batch_size: int = 8) -> Generator[Dict[str, Any], None, None]:
            """
            Streams items, but performs selection/context/LLM disambiguation in batches.
            """
            query = f"""
            MATCH (n:{self.source_label})
            WHERE NOT "ProcessedWithOntologyMapper" IN labels(n)
            RETURN n.id AS id, n.label AS label
            """
            rows: List[Tuple[str, str]] = []
            with self._driver.session(database=self._database) as session:
                for rec in session.run(query):
                    rows.append((rec["id"], rec["label"]))

            # chunk and process
            for i in range(0, len(rows), batch_size):
                chunk = rows[i:i + batch_size]  # List[(id, label)]
                results = self.run_disambiguation_in_batch(chunk, max_concurrency=8)

                for (sid, lbl, md) in results:
                    if md.confidence >= self.llm_threshold:
                        yield {
                            "id": sid,
                            "label": lbl,
                            "disambiguation_result": self.md_to_param(md),
                        }
    
        def merge_mapping_relationship(self):
            logging.info(f"Merging mapping relationship: {self.source_label} -> {self.target_label} [{self.relationship_type}]")
            query = f"""
            UNWIND $batch AS item
            MATCH (source:{self.source_label} {{id: item.id}})
            MATCH (target:{self.target_label} {{id: item.disambiguation_result.best_id}})
            MERGE (source)-[r:{self.relationship_type}]->(target)
            SET r.{self.relationship_conf_prop} = item.disambiguation_result.confidence,
                r.rationale = item.disambiguation_result.rationale,
                r.support = item.disambiguation_result.support
            SET source:ProcessedWithOntologyMapper
            """
            size = self.count_missing_source_nodes()
            # get_source_nodes() now performs batch disambiguation under the hood
            self.batch_store(query, self.get_source_nodes_in_batch(), size=size)

        def apply_updates(self):
            logging.info("Testing Ontology Mapper...")
            self.test()
            print("\n\n\n")
            logging.info("Starting Ontology Mapping import...")
            self.merge_mapping_relationship()

    return OntologyMapper


if __name__ == '__main__':
    from util.cli_entry import run_backend_importer

    run_backend_importer(
        ontology_mapper_factory,
        description="Run Ontology Mapper.",
        file_help="No file needed for ontology mapping.",
        default_base_path="./data/",
        require_file=False
    )
