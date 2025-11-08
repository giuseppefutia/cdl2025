from langchain_core.tools import tool

from llm.pydantic_model import OntologyMappingInput

from llm.chain import ontology_mapping_chain

def build_ontology_mapper_tool(llm):
    chain = ontology_mapping_chain(llm)

    @tool("ontology_mapper", args_schema=OntologyMappingInput)
    def ontology_mapper_tool(source_concept: str, 
                             source_context: str,
                             candidate_list: str):
        """ Map a source medical concept to the best target candidate and return a structured result. """
        result = chain.invoke({
            "source_concept": source_concept,
            "source_context": source_context,
            "candidate_list": candidate_list,
        })
        return result.model_dump()

    return ontology_mapper_tool