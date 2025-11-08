from __future__ import annotations

import logging
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from neo4j.exceptions import ClientError as Neo4jClientError

from llm.tool import build_ontology_mapper_tool
from util.config_loader import load_config_api
from util.api_client import ApiClient

def ontology_mapper_factory(base_importer_cls, backend: str, config_path: str = "config.ini"):

    class EmbedAPI:
        def __init__(self, api):
            self.api = api

        def embed(self, text: str) -> List[float]:
            # returns a single embedding vector
            r = self.api.post('/embed', {'input': [text]})
            return r[0]['data'][0]

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
            self.cfg_emb = load_config_api("embedding", path=config_path)
            self.emb_api = EmbedAPI(ApiClient(self.cfg_emb))

            # LLM Model Initialization
            url_llm = load_config_api("llm", path=config_path)
            self.llm = ChatOpenAI(
                api_key="EMPTY",
                base_url=url_llm,
                model_name="google/medgemma-4b-it",
                temperature=0,
            )
            self.ontology_mapper_tool = build_ontology_mapper_tool(self.llm)
            
            # Configuration parameters
            self.source_label: str = "IcdDisease"
            self.target_label: str = "HpoPhenotype"
            self.target_index_name: str = "hpo_phenotype_embedding"
            self.relationship_type: str = "ICD_MAPS_TO_HPO_PHENOTYPE"
            self.relationship_conf_prop: str = "confidence"
            self.llm_threshold: float = 0.7
            self.k: int = 10
            self.write_relationships: bool = True
            self.mapper_task : str = "icd_to_hpo_phenotype"
        
        def md_to_param(self, md: MappingDecision) -> Dict[str, Any]:
            """ Convert MappingDecision dataclass to dict for Neo4j parameters. """
            d = asdict(md)                               # dataclass -> plain dict
            d["support"] = json.dumps(d.get("support", {}), ensure_ascii=False)  # map -> JSON string
            return {k: v for k, v in d.items() if v is not None} 
        
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
        
        def build_context(self, source_id: str, candidates: List[Candidate]) -> str:
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

        def disambiguate_candidates(self, source_concept: str, source_context: str, candidates: List[Candidate]) -> MappingDecision:            
            payload = {
                "source_concept": source_concept,
                "source_context": json.dumps(source_context),
                "candidate_list": json.dumps(candidates)
            }
            response = self.ontology_mapper_tool.invoke(payload)
            try:
                best_id = response.get("best_id")
                best_label = response.get("best_label")
                confidence = response.get("confidence", 0.0)
                rationale = response.get("rationale", "")
                support = response.get("support", {})
                return MappingDecision(
                    best_id=best_id,
                    best_label=best_label,
                    confidence=confidence,
                    rationale=rationale,
                    support=support
                )
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse LLM response: {e}")
                return MappingDecision(
                    best_id=None,
                    best_label=None,
                    confidence=0.0,
                    rationale="Failed to parse LLM response.",
                    support={}
                )
        
        def run_disambiguation(self, source_id, source_label: str):
            logging.debug(f"Running disambiguation for source_id: {source_id}, source_label: {source_label}")
            candidates = self.select_candidates(source_label)
            logging.debug(f"Selected {len(candidates)} candidates for source_label: {source_label}")
            context = self.build_context(source_id=source_id, candidates=candidates)
            logging.debug(f"Built context for source_id: {source_id}")
            source_context = context.get("source", {})
            logging.debug(f"Source context: {source_context}")
            candidate_contexts = context.get("candidates", [])
            logging.debug(f"Candidate contexts: {candidate_contexts}")
            res_disambiguation = self.disambiguate_candidates(
                source_concept=source_label,
                source_context=source_context,
                candidates=candidate_contexts
            )
            logging.debug(f"Disambiguation result: {res_disambiguation}")
            return res_disambiguation
        
        def test(self):
            source_label = "Syncope and collapse"
            candidates = self.select_candidates(source_label)
            print(f"Top {len(candidates)} candidates for source '{source_label}':")
            for candidate in candidates:
                print(f" - {candidate.label} (ID: {candidate.id}, Score: {candidate.score})")
            
            context = self.build_context(source_id="R55", candidates=candidates)
            source_context = context.get("source", {})
            candidate_contexts = context.get("candidates", [])
            
            self.disambiguate_candidates(
                source_concept=source_label,
                source_context=source_context,
                candidates=candidate_contexts
            )
        
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
        
        def merge_mapping_relationship(self):
            print(self.source_label, self.target_label, self.relationship_type)
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
            self.batch_store(query, self.get_source_nodes(), size=size)
            
        def apply_updates(self):
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