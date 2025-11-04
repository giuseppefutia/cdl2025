import logging
import time
from typing import Sequence, List

from neo4j.exceptions import ClientError as Neo4jClientError

from util.config_loader import load_config_api
from util.api_client import ApiClient


def hpo_embedding_factory(base_importer_cls: str, backend: str, config_path: str = "config.ini"):

    class HPOEmbeddingImporter(base_importer_cls):

        def __init__(self):
            super().__init__()
            self.backend = backend
            self.cfg = load_config_api("embedding", path=config_path)
            self.api = ApiClient(self.cfg)

        def set_index(self):
            queries = ["""
                CREATE VECTOR INDEX hpo_phenotype_embedding IF NOT EXISTS
                FOR (n:HPOPhenotype)
                ON (n.embedding_label)
                OPTIONS {
                    IndexConfig: {
                        `vector.dimensions`: 3584,
                        `vector.similarity_function`: 'cosine'
                    }
                };
                """]
            with self._driver.session(database=self._database) as session:
                for q in queries:
                    try:
                        session.run(q)
                    except Neo4jClientError as e:
                        # ignore if we already have the rule in place
                        if e.code != "Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists":
                            raise e

        def _embed_label(self, text: str) -> Sequence[float]:
            """
            Single-text helper (kept for compatibility).
            """
            resp = self.api.post('/embed', {'input': text})
            data = resp[0]["data"][0]  # your proxy returns the vector directly
            return data

        def _embed_labels(self, texts: List[str]) -> List[Sequence[float]]:
            """
            Batched helper: call the proxy once with many inputs.
            Assumes your proxy returns: resp[0]["data"] -> list of vectors in order.
            """
            if not texts:
                return []
            resp = self.api.post('/embed', {'input': texts})
            vectors = resp[0]["data"]
            if len(vectors) != len(texts):
                raise RuntimeError(f"Embedding count mismatch ({len(vectors)} != {len(texts)})")
            return vectors

        def count_hpo_phen_label_missing_embeddings(self) -> int:
            query = """
            MATCH (n:HpoPhenotype)
            WHERE n.embedding_label IS NULL
            RETURN count(n) AS cnt
            """
            with self._driver.session(database=self._database) as session:
                result = session.run(query)
                record = result.single()
                return record["cnt"] if record else 0

        def add_phen_label_embeddings(self, batch_size: int = 128):
            """
            Embed missing HPO phenotype labels in batches and write with UNWIND.
            Logs progress (processed/total, %, rate, ETA) after each batch.
            """
            fetch_query = """
            MATCH (n:HpoPhenotype)
            WHERE n.embedding_label IS NULL AND n.label IS NOT NULL
            RETURN n.id AS id, n.label AS label
            """

            write_query = """
            UNWIND $rows AS row
            MATCH (n:HpoPhenotype {id: row.id})
            SET n.embedding_label = row.embedding_label
            """

            total = self.count_hpo_phen_label_missing_embeddings()
            if total == 0:
                logging.info("No missing embeddings found. Nothing to do.")
                return

            processed = 0
            start_ts = time.time()
            batch_idx = 0

            def log_progress():
                elapsed = max(time.time() - start_ts, 1e-6)
                rate = processed / elapsed  # nodes/sec
                pct = (processed / total) * 100 if total else 100.0
                remaining = max(total - processed, 0)
                eta = remaining / rate if rate > 0 else float('inf')
                # Keep it short & simple
                logging.info(
                    f"[HPO-Embed] Batch {batch_idx} | "
                    f"{processed}/{total} ({pct:.1f}%) | "
                    f"{rate:.1f} nodes/s | ETA ~ {int(eta)}s"
                )

            with self._driver.session(database=self._database) as session:
                result = session.run(fetch_query)

                buffer_ids: List[str] = []
                buffer_labels: List[str] = []

                def flush():
                    nonlocal processed, batch_idx
                    if not buffer_labels:
                        return
                    embeddings = self._embed_labels(buffer_labels)
                    rows = [{"id": i, "embedding_label": e} for i, e in zip(buffer_ids, embeddings)]
                    session.run(write_query, rows=rows)

                    processed += len(rows)
                    batch_idx += 1
                    buffer_ids.clear()
                    buffer_labels.clear()
                    log_progress()

                for rec in result:
                    node_id = rec["id"]
                    label = rec["label"]
                    if not label:
                        continue
                    buffer_ids.append(node_id)
                    buffer_labels.append(label)
                    if len(buffer_labels) >= batch_size:
                        flush()

                flush()  # remainder

            # Final summary
            elapsed = max(time.time() - start_ts, 1e-6)
            logging.info(f"[HPO-Embed] Completed: {processed}/{total} in {int(elapsed)}s "
                         f"({processed/elapsed:.1f} nodes/s)")

        def apply_updates(self):
            logging.info("Loading index...")
            self.set_index()

            logging.info("Adding HPO phenotype label embeddings...")
            self.add_phen_label_embeddings()

    return HPOEmbeddingImporter


if __name__ == '__main__':
    from util.cli_entry import run_backend_importer

    run_backend_importer(
        hpo_embedding_factory,
        description="Run HPO Embedding.",
        file_help="No file needed for HPO embedding.",
        default_base_path="./data/",
        require_file=False
    )
