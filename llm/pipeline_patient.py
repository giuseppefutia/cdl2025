import ast
from datetime import datetime, date
from typing import Any, Dict, Optional, List

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


def _parse_encounter_start(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        # Handles 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS'
        return datetime.fromisoformat(value).date()
    except Exception:
        try:
            # Fallback: take the first 10 chars as YYYY-MM-DD
            return datetime.strptime(value[:10], "%Y-%m-%d").date()
        except Exception:
            return None


def get_patient_views(
    patient_id: str,
    encounter_date: Optional[date] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch virtualized 'patient' resources via APOC DV and return a list of
    normalized views.
    Args:
        patient_id: The PatientId to query.
        encounter_date: If provided, filter to only include encounters
            with this exact Encounter.period.start date.
    Returns:
        A list of patient views as dictionaries.
    """
    cypher = """
    CALL apoc.dv.query('patient', {patientId: $patient_id}) YIELD node AS v
    RETURN v
    """
    rows = enhanced_graph.query(cypher, {"patient_id": patient_id})
    if not rows:
        return []

    views: List[Dict[str, Any]] = []

    for row in rows:
        v = row.get("v")

        if isinstance(v, dict) and "properties" in v:
            props = v.get("properties", {})
            identity = v.get("identity")
            labels = v.get("labels", [])
            element_id = v.get("elementId") or v.get("element_id")
        else:
            props = v if isinstance(v, dict) else {}
            identity = None
            labels = []
            element_id = None

        # --- date filtering logic ---
        encounter_start_str = props.get("Encounter.period.start")
        encounter_start_date = _parse_encounter_start(encounter_start_str)

        if encounter_date is not None:
            # Strict: only keep rows we can parse AND that match the requested date
            if encounter_start_date is None or encounter_start_date != encounter_date:
                continue  # skip this row

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
                "period_start": encounter_start_str,
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

            # Traceability
            "filters": {
                "encounter_date_requested": (
                    encounter_date.isoformat() if encounter_date else None
                ),
                "encounter_start_date_parsed": (
                    encounter_start_date.isoformat()
                    if encounter_start_date
                    else None
                ),
            },
        }

        views.append(view)

    return views

