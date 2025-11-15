from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm.prompt import (
    DIAGNOSE_CYPHER_PROMPT,
    ONTOLOGY_MAPPING_PROMPT,
    PATIENT_NER_PROMPT,
    PATIENT_NED_PROMPT,
    TEXT_2_CYPHER_PROMPT,
    QUERY_VALIDATION_PROMPT,
    QUERY_CORRECTION_PROMPT,
)

from llm.pydantic_model import (
    DiagnoseCypherOutput,
    OntologyMappingResponse,
    PatientNERResponse,
    PatientNEDResponse,
    ValidateCypherOutput,
    GeneralMedicalInput,
)

#######################
### Building chains ###
#######################


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


########################
### Retrieval chains ###
########################


def text2cypher_chain(llm_model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            TEXT_2_CYPHER_PROMPT["system"],
            template_format="jinja2",
        ),
        HumanMessagePromptTemplate.from_template(
            TEXT_2_CYPHER_PROMPT["user"],
            template_format="jinja2",
        ),
    ])
    return prompt | llm_model | StrOutputParser()


def validate_cypher_chain(llm_model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            QUERY_VALIDATION_PROMPT["system"],
            template_format="jinja2",
        ),
        HumanMessagePromptTemplate.from_template(
            QUERY_VALIDATION_PROMPT["user"],
            template_format="jinja2"
        )
    ])
    return prompt | llm_model.with_structured_output(ValidateCypherOutput)


def diagnose_cypher_chain(llm_model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            DIAGNOSE_CYPHER_PROMPT["system"],
            template_format="jinja2",
        ),
        HumanMessagePromptTemplate.from_template(
            DIAGNOSE_CYPHER_PROMPT["user"],
            template_format="jinja2",
        ),
    ])
    # 👇 this is where the Pydantic model is actually used
    return prompt | llm_model.with_structured_output(DiagnoseCypherOutput)


def correct_cypher_chain(llm_model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            QUERY_CORRECTION_PROMPT["system"],
            template_format="jinja2",
        ),
        HumanMessagePromptTemplate.from_template(
            QUERY_CORRECTION_PROMPT["user"],
            template_format="jinja2"
        )
    ])
    return prompt | llm_model | StrOutputParser()

