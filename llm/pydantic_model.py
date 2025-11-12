
from typing import List, Optional
from pydantic import BaseModel, Field

###############################
### Ontology Mapping Models ###
###############################

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

##########################
### NER Patient Models ###
##########################

class PatientNERInput(BaseModel):
    icd_chapters: List[str]
    patient_id: str
    encounter_id: str
    concat_text: str
    narrative_text: str

class PatientNEREntity(BaseModel):
    source: str        # "concat" or "narrative"
    start: int         # 0-based [start, end)
    end: int
    text: str
    label: str         # must be one of icd_chapters
    assertion: str     # "present" | "negated" | "uncertain"
    temporality: str   # "acute" | "chronic" | "recurrent" | "history" | "unspecified"
    rationale: str

class PatientNERResponse(BaseModel):
    patient_id: str
    encounter_id: str
    entities: List[PatientNEREntity] = Field(default_factory=list)

###################
### Patient NED ###
###################

class PatientNEDCandidate(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    id: str                 # e.g., "F32.0"
    label: str              # e.g., "Mild depressive episode"

class PatientNEDOtherMention(BaseModel):
    text: str
    label: str              # ICD Chapter label for the other mention

class PatientNEDInput(BaseModel):
    mention: PatientNEREntity
    candidates: List[PatientNEDCandidate]
    other_mentions: List[PatientNEDOtherMention] = Field(default_factory=list)

class PatientNEDResponse(PatientNEREntity):
    icd_id: Optional[str] = None        # e.g., "R45.2"; None if abstaining
    icd_label: Optional[str] = None     # e.g., "Unhappiness"; None if abstaining
    confidence: float = Field(default=None, ge=0.0, le=1.0)
    linking_rationale: str