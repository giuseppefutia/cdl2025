import ast
from typing import Any, Dict, Optional

from langchain_neo4j import Neo4jGraph

enhanced_graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password",
    database="cdl2025",
    enhanced_schema=True,
)


def _parse_python_list_string(value: Optional[str]):
    """Parse stringified Python lists like "['G57.1', ...]" or "[{...}, ...]" safely."""
    if not value or not isinstance(value, str):
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return None


def get_patient_view(patient_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a virtualized 'patient' resource via APOC DV and
    return a normalized view tailored for the LLM explanation prompt.
    """
    cypher = """
    CALL apoc.dv.query('patient', {patientId: $patient_id}) YIELD node AS v
    RETURN v
    """
    rows = enhanced_graph.query(cypher, {"patient_id": patient_id})
    if not rows:
        return None

    v = rows[0]["v"]

    # With enhanced_schema=True, v is typically a dict like your sample:
    # { "identity": ..., "labels": [...], "properties": {...}, "elementId": ... }
    if isinstance(v, dict) and "properties" in v:
        props = v.get("properties", {})
        identity = v.get("identity")
        labels = v.get("labels", [])
        element_id = v.get("elementId") or v.get("element_id")
    else:
        # Fallback: try to treat v as a bare properties dict
        props = v if isinstance(v, dict) else {}
        identity = None
        labels = []
        element_id = None

    icd_codes = _parse_python_list_string(props.get("ICD10_Codes"))
    ner_entities = _parse_python_list_string(props.get("NER_Entities"))
    ned_entities = _parse_python_list_string(props.get("NED_Entities"))

    view: Dict[str, Any] = {
        "patient_id": props.get("PatientId") or props.get("patientId"),
        "identity": identity,
        "labels": labels,
        "elementId": element_id,

        # High-level clinical fields
        "condition": props.get("Condition"),
        "chief_complaint": props.get("ChiefComplaint"),
        "course_trend": props.get("CourseTrend"),
        "comorbidities": props.get("Comorbidities") or "",
        "plan_followup": props.get("Plan/FollowUp"),
        "medication_statement": props.get("MedicationStatement"),
        "notes": props.get("Notes"),

        # Encounter info
        "encounter": {
            "id": props.get("EncounterID"),
            "period_start": props.get("Encounter.period.start"),
            "reason_code": props.get("Encounter.reasonCode"),
            "class": props.get("Encounter.class"),
            "discharge_disposition": props.get(
                "Encounter.hospitalization.dischargeDisposition"
            ),
            "diagnosis_rank": props.get("Encounter.diagnosis.rank"),
        },

        # Narrative + observations
        "narrative": props.get("Narrative"),
        "observation_vitals": props.get("Observation[vitals]"),
        "observation_text": props.get("Observation[key]"),
        "diagnostic_report": props.get("DiagnosticReport"),
        "procedure": props.get("Procedure"),

        # NLP/ICD-related
        "icd10_codes": icd_codes,
        "ner_entities": ner_entities,
        "ned_entities": ned_entities,
    }

    return view
