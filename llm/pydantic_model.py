
from typing import List, Optional
from pydantic import BaseModel, Field

###########################################
### Ontology Mapping LLM Pydantic Models ##
###########################################

class OntologyMappingInput(BaseModel):
    source_concept: str
    source_context: str
    candidate_list: str

class SupportItem(BaseModel):
    evidence: str
    reason: str

class OntologyMappingResponse(BaseModel):
    best_id: str
    best_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    support: SupportItem