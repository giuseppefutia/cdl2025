from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI

from llm.prompt import (
    ONTOLOGY_MAPPING_PROMPT,
    PATIENT_NER_PROMPT,
    PATIENT_NED_PROMPT,
)

from llm.pydantic_model import (
    OntologyMappingResponse,
    PatientNERResponse,
    PatientNEDResponse,
)

def ontology_mapping_chain(llm_model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            ONTOLOGY_MAPPING_PROMPT["system"],
            template_format="jinja2",
        ),
        HumanMessagePromptTemplate.from_template(
            ONTOLOGY_MAPPING_PROMPT["user"],
            template_format="jinja2",
        ),
    ])
    return prompt | llm_model.with_structured_output(OntologyMappingResponse)

def patient_ner_chain(llm_model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            PATIENT_NER_PROMPT["system"],
            template_format="jinja2",
        ),
        HumanMessagePromptTemplate.from_template(
            PATIENT_NER_PROMPT["user"],
            template_format="jinja2",
        ),
    ])
    return prompt | llm_model.with_structured_output(PatientNERResponse)

def patient_ned_chain(llm_model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            PATIENT_NED_PROMPT["system"],
            template_format="jinja2",
        ),
        HumanMessagePromptTemplate.from_template(
            PATIENT_NED_PROMPT["user"],
            template_format="jinja2",
        ),
    ])
    return prompt | llm_model.with_structured_output(PatientNEDResponse)
