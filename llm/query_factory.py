import time

from typing import List, Dict, Any, Optional

from langchain_neo4j import Neo4jGraph
from neo4j.exceptions import ServiceUnavailable, SessionExpired

class QueryFragmentFactory:
    """Auto-generates the Cypher *fragments* per step, using the live schema
    when possible. Falls back to safe templates if needed.
    """

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        self.schema = graph.structured_schema

    # --- helpers ---
    def _rel_exists(self, start: str, rel: str, end: str) -> bool:
        for r in self.schema.get("relationships", []):
            if r.get("start") == start and r.get("type") == rel and r.get("end") == end:
                return True
        return False

    # --- fragment makers ---
    def patient_icd_fragment(self) -> str:
        # Keep BOM-safe fallback for property
        return (
            "CALL apoc.dv.query('patient', {patientId:$pid}) YIELD node AS v"
            "WITH v, apoc.convert.fromJsonList(coalesce(v.ICD10_Codes, v.`\uFEFFICD10_Codes`)) AS codes"
            "UNWIND codes AS code"
        )

    def icd_to_hpo_fragment(self) -> str:
        rel = ":ICD_MAPS_TO_HPO_PHENOTYPE" if self._rel_exists("IcdDisease", "ICD_MAPS_TO_HPO_PHENOTYPE", "HpoPhenotype") else ":ICD_MAPS_TO_HPO_PHENOTYPE"
        return (
            "MATCH (icd:IcdDisease {id: code})"
            f"MATCH (icd){rel}->(h:HpoPhenotype)"
            "WITH collect(DISTINCT h) AS phenos"
        )

    def rollup_fragment(self) -> str:
        return (
            "UNWIND phenos AS h"
            "MATCH (h)-[:SUBCLASSOF*0..]->(anc:HpoPhenotype)"
            "WITH apoc.coll.toSet(collect(DISTINCT anc.id)) AS target"
            "WITH [id IN target WHERE id IS NOT NULL] AS target"
        )

    def coverage_fragment(self) -> str:
        rel = ":HAS_PHENOTYPIC_FEATURE" if self._rel_exists("HpoDisease", "HAS_PHENOTYPIC_FEATURE", "HpoPhenotype") else ":HAS_PHENOTYPIC_FEATURE"
        return (
            "MATCH (d:HpoDisease)" + rel + "->(dh:HpoPhenotype)"
            "WHERE dh.id IN target"
            "WITH d, target, collect(DISTINCT dh.id) AS dSet"
            "WITH d, target, apoc.coll.toSet(dSet) AS got"
            "WITH d, target, apoc.coll.intersection(target, got) AS overlap,"
            "     apoc.coll.subtract(target, got) AS missing"
            "WITH d, size(overlap) AS covered, size(target) AS total, missing"
            "WHERE covered > 0"
            "RETURN d.id AS diseaseId, d.label AS diseaseName, covered, total,"
            "       round(100.0 * covered / total, 1) AS coveragePct, missing AS missingHpoIds"
            "ORDER BY coveragePct DESC, covered DESC"
            "LIMIT $limit"
        )

    def stitched_query(self) -> str:
        return "".join([
            self.patient_icd_fragment(),
            self.icd_to_hpo_fragment(),
            self.rollup_fragment(),
            self.coverage_fragment(),
        ])

def get_patient_icd_codes(patient_id: str) -> List[str]:
    """Fetch ICD10 codes for a patient, handling BOM-prefixed property name.

    Returns a list of strings (ICD-10 codes).
    """
    cypher = """
        CALL apoc.dv.query('patient', {patientId:'P001'}) YIELD node AS v
        RETURN v,
        apoc.convert.fromJsonList(
        coalesce(apoc.any.property(v,'ICD10_Codes'),
                    apoc.any.property(v,'\uFEFFICD10_Codes'))
        ) AS codes
    """
    rows = _run_query(cypher, {"pid": patient_id})
    if not rows:
        return []
    codes = rows[0].get("codes") or []
    # Ensure unique, uppercase, non-empty
    clean = sorted({(c or "").strip().upper() for c in codes if c})
    return clean


def map_icd_to_hpo(icd_codes: List[str]) -> List[str]:
    """Map ICD codes to HPO phenotype IDs via :ICD_MAPS_TO_HPO_PHENOTYPE."""
    if not icd_codes:
        return []
    cypher = (
        "UNWIND $codes AS code\n"
        "MATCH (:IcdDisease {id: code})-[:ICD_MAPS_TO_HPO_PHENOTYPE]->(h:HpoPhenotype)\n"
        "RETURN collect(DISTINCT h.id) AS hpo_ids"
    )
    rows = _run_query(cypher, {"codes": icd_codes})
    return rows[0].get("hpo_ids", []) if rows else []


def rollup_hpo_to_ancestors(hpo_ids: List[str]) -> List[str]:
    """Roll up to ancestors (including self) over :SUBCLASSOF*0.. and return unique IDs."""
    if not hpo_ids:
        return []
    cypher = (
        "UNWIND $ids AS hid\n"
        "MATCH (h:HpoPhenotype {id: hid})-[:SUBCLASSOF*0..]->(anc:HpoPhenotype)\n"
        "WITH collect(DISTINCT anc.id) AS target\n"
        "RETURN [id IN apoc.coll.toSet(target) WHERE id IS NOT NULL] AS target"
    )
    rows = _run_query(cypher, {"ids": hpo_ids})
    return rows[0].get("target", []) if rows else []


def compute_coverage(target_ids: List[str], limit: int = 20) -> List[Dict[str, Any]]:
    """Given rolled-up HPO target IDs, compute coverage across diseases."""
    if not target_ids:
        return []
    cypher = (
        "MATCH (d:HpoDisease)-[:HAS_PHENOTYPIC_FEATURE]->(dh:HpoPhenotype)\n"
        "WHERE dh.id IN $target\n"
        "WITH d, apoc.coll.toSet(collect(DISTINCT dh.id)) AS got, $target AS target\n"
        "WITH d, target, apoc.coll.intersection(target, got) AS overlap,\n"
        "     apoc.coll.subtract(target, got) AS missing\n"
        "WITH d, size(overlap) AS covered, size(target) AS total, missing\n"
        "WHERE covered > 0\n"
        "RETURN d.id AS diseaseId, d.label AS diseaseName, covered, total,\n"
        "       round(100.0 * covered / total, 1) AS coveragePct, missing AS missingHpoIds\n"
        "ORDER BY coveragePct DESC, covered DESC\n"
        "LIMIT $limit"
    )
    return _run_query(cypher, {"target": target_ids, "limit": int(limit)})

enhanced_graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password",
    database="cdl2025",
    enhanced_schema=True,
)

def _run_query(cypher: str, params: Optional[Dict[str, Any]] = None, retries: int = 2):
    """Run a Cypher query with minimal retry on transient errors."""
    params = params or {}
    attempt = 0
    while True:
        try:
            return enhanced_graph.query(cypher, params)
        except (ServiceUnavailable, SessionExpired) as e:
            if attempt >= retries:
                raise
            attempt += 1
            time.sleep(0.5 * attempt)
        except Exception:
            raise