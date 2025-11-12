from typing import List
from langchain_core.tools import tool
import json

from llm.pydantic_model import (
    OntologyMappingInput,
    PatientNERInput,
    PatientNEREntity,
    PatientNEDInput,
    PatientNEDCandidate,
    PatientNEDOtherMention,
)
from llm.chain import (
    ontology_mapping_chain,
    patient_ner_chain,
    patient_ned_chain,
)

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
