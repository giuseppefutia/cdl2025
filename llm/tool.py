import json
from typing import List

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from llm.prompt import NEO4J_SCHEMA

from llm.pydantic_model import (
    OntologyMappingInput,
    PatientNERInput,
    PatientNEREntity,
    PatientNEDInput,
    PatientNEDCandidate,
    PatientNEDOtherMention,
    GeneralMedicalInput,
    GeneralMedicalResponse,
    PatientCoverageInput,
    CoverageRow,
    PatientCoverageResponse,
)
from llm.chain import (
    ontology_mapping_chain,
    patient_ner_chain,
    patient_ned_chain,
    clinician_explanation_chain,
    patient_coverage_chain,
)

from llm.pipeline import text2cypher_pipeline, enhanced_graph
from llm.query_factory import QueryFragmentFactory

### SHOULD BE MOVED SOME WHERE ELSE LATER ###

fragment_factory = QueryFragmentFactory(enhanced_graph)

def build_auto_query() -> str:
    """Return the full step-by-step Cypher assembled automatically from fragments."""
    return fragment_factory.stitched_query()

def run_auto_query(patient_id: str, limit: int = 20):
    cypher = build_auto_query()
    return enhanced_graph.query(cypher, {"pid": patient_id, "limit": int(limit)})


########################

def build_ontology_mapper_tool(llm):
    chain = ontology_mapping_chain(llm)

    @tool("ontology_mapper", args_schema=OntologyMappingInput)
    def ontology_mapper_tool(
        source_concept: str,
        source_context: str,
        candidate_list: str
    ):
        """Map a source medical concept to the best target candidate and return a structured result."""
        result = chain.invoke({
            "source_concept": source_concept,
            "source_context": source_context,
            "candidate_list": candidate_list,
        })
        return result.model_dump()

    return ontology_mapper_tool


def build_patient_ner_tool(llm):
    chain = patient_ner_chain(llm)

    @tool("patient_ner", args_schema=PatientNERInput)
    def patient_ner_tool(
        icd_chapters: List[str],
        patient_id: str,
        encounter_id: str,
        concat_text: str,
        narrative_text: str
    ):
        """Annotate clinical text with ICD Chapter labels and return a structured result."""
        result = chain.invoke({
            "icd_chapters": icd_chapters,
            "patient_id": patient_id,
            "encounter_id": encounter_id,
            "concat_text": concat_text,
            "narrative_text": narrative_text,
        })
        return result.model_dump()

    return patient_ner_tool


def build_patient_ned_tool(llm):
    chain = patient_ned_chain(llm)

    @tool("patient_ned", args_schema=PatientNEDInput)
    def patient_ned_tool(
        mention: PatientNEREntity,
        candidates: list[PatientNEDCandidate],
        other_mentions: list[PatientNEDOtherMention] = [],
    ):
        """Disambiguate a medical mention to the best ICD code candidate and return a structured result."""
        result = chain.invoke({
            "mention": mention,
            "candidates": candidates,
            "other_mentions": other_mentions,
        })
        return result.model_dump()

    return patient_ned_tool


def build_general_medical_tool(llm: ChatOpenAI, debug: bool = False):
    explanation_chain = clinician_explanation_chain(llm)

    @tool("general_medical_executor", args_schema=GeneralMedicalInput)
    def general_medical_executor(
        question: str,
        top_k: int = 20,
    ):
        """
        Run a general medical query against the Neo4j knowledge graph, converting
        the question to Cypher, truncating results to top_k, and returning rows
        along with an LLM-generated explanation.
        """
        cypher, rows = text2cypher_pipeline(llm, question, debug=debug)

        if isinstance(rows, list) and top_k is not None:
            rows = rows[: int(top_k)]

        # Prepare JSON for the prompt
        rows_json = json.dumps(rows, default=str)

        try:
            explanation = explanation_chain.invoke(
                {
                    "schema": NEO4J_SCHEMA.strip(),
                    "question": question,
                    "cypher": cypher,
                    "rows_json": rows_json,
                }
            )
        except Exception as e:
            explanation = f"Explanation could not be generated: {e}"

        return GeneralMedicalResponse(
            cypher=cypher,
            rows=rows,
            steps=["text2cypher", "validated", "executed", "explained"],
            explanation=explanation,
        ).model_dump()

    return general_medical_executor


def build_patient_coverage_tool(llm: ChatOpenAI):
    chain = patient_coverage_chain(llm)

    @tool("patient_coverage")
    def patient_coverage(
        patient_id: str,
        limit: int = 20):
        """Deterministic single-patient coverage using fragment composer."""
        _ = chain.invoke({"patient_id": patient_id, "limit": int(limit)}) # for auditability
        cypher = build_auto_query()
        rows = enhanced_graph.query(cypher, {"pid": patient_id, "limit": int(limit)})
        # Cast rows into CoverageRow when possible
        cast_rows: List[CoverageRow] = []
        for r in rows or []:
            try:
                cast_rows.append(CoverageRow(**r))
            except Exception:
                pass
            return PatientCoverageResponse(cypher=cypher, rows=cast_rows or rows, steps=["fragments", "stitched", "executed"]).model_dump()

    return patient_coverage
