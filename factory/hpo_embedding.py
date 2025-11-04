import logging
from typing import Sequence

from neo4j.exceptions import ClientError as Neo4jClientError

from util.config_loader import load_config_api
from util.api_client import ApiClient


def hpo_embedding_factory(base_importer_cls:str, backend:str, config_path: str = "config.ini"):

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
            Helper that calls the embeddings proxy defined at [api].uri_embedding.
            Uses the mean endpoint to get a single vector for the string.
            """
            resp = self.api.post('/embed', {'input': text})
            data = resp[0]["data"][0]
            return data
        
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

        def get_hpo_phen_label_embeddings(self):
            query = """
            MATCH (n:HpoPhenotype)
            WHERE n.embedding_label IS NULL
            RETURN n
            """
            with self._driver.session(database=self._database) as session:
                result = session.run(query)
                for record in result:
                    node = record["n"]
                    label = node.get("label")
                    if not label:
                        continue
                    embedding = self._embed_label(label)
                    yield {
                        "id": node.get("id"),
                        "embedding_label": embedding
                    }
            
        def add_phen_label_embeddings(self):
            query = """
            MATCH (n:HpoPhenotype {id: $id})
            SET n.embedding_label = $embedding_label
            """
            size = self.count_hpo_phen_label_missing_embeddings()
            self.batch_store(query, self.get_hpo_phen_label_embeddings(), size=size)     
        
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
        require_file = False
    )