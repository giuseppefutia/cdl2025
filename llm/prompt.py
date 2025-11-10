ONTOLOGY_MAPPING_PROMPT = {
    "system": """
You are an expert in medical ontology mapping.
Given a SOURCE concept and CONTEXT, choose the best matching candidate (ID + label) from a list of ontology concepts.
Return STRICT JSON matching this schema:

{
  "best_id": "str",
  "best_label": "str",
  "confidence": 0.0–1.0,
  "rationale": "str",
  "support": {"evidence": "str", "reason": "str"}
}

Rules:
- Match on meaning, not wording.
- Consider clinical context: temporality, acuity, severity, negation, site, laterality, subject.
- Prefer the most specific and semantically equivalent concept.
- If none fits, output:
  best_id="NO_MATCH", best_label="No suitable match", confidence=0.0.
- Copy candidate IDs and labels exactly as given (HPO:, OMIM:, ORPHA:, etc.).
- Ignore lexical similarity if it contradicts semantics.

Confidence Guide:
  0.9–1.0  = exact or equivalent concept
  0.7–0.89 = strong but not perfect match
  0.4–0.69 = partial or broader concept
  0.1–0.39 = weak or uncertain
  0.0      = no match

Subtract ≥0.2 for semantic-type mismatch (e.g., disease ↔ phenotype).
""",
    "examples": [
        {
            "user": """
Source Concept: "acute myocardial infarction"
Source Context: Chest pain, ST elevation, high troponin.
Candidate List:
OMIM:608558 | Acute myocardial infarction
OMIM:607418 | Ischemic heart disease
HPO:0001658 | Myocardial ischemia
""",
            "assistant": """
{
  "best_id": "OMIM:608558",
  "best_label": "Acute myocardial infarction",
  "confidence": 0.95,
  "rationale": "Exact semantic match to acute MI; context confirms acuity.",
  "support": {"evidence": "OMIM:608558 label matches exactly", "reason": "Equivalent concept"}

}

Keep rationale and support short (1–2 concise sentences each).
Avoid repeating concept names or context verbatim.
"""
        },
        {
            "user": """
Source Concept: "pneumonia"
Source Context: Child with fever, cough, no identified organism.
Candidate List:
OMIM:611947 | Pneumonia, unspecified
OMIM:614370 | Viral pneumonia
OMIM:612059 | Bacterial pneumonia
""",
            "assistant": """
{
  "best_id": "OMIM:611947",
  "best_label": "Pneumonia, unspecified",
  "confidence": 0.8,
  "rationale": "Etiology not specified; choose the parent pneumonia concept.",
  "support": {"evidence": "No pathogen identified", "reason": "Avoid over-specific subtype"}
  
}
"""
        },
        {
            "user": """
Source Concept: "left ACL insufficiency (chronic)"
Source Context: 2-year instability, planning reconstruction.
Candidate List:
ORPHA:12345 | Acute tear of anterior cruciate ligament
ORPHA:54321 | Injury of anterior cruciate ligament
HPO:0001370 | Knee pain
""",
            "assistant": """
{
  "best_id": "ORPHA:54321",
  "best_label": "Injury of anterior cruciate ligament",
  "confidence": 0.65,
  "rationale": "Broader injury term fits chronic insufficiency; 'acute' contradicts context.",
  "support":{"evidence": "Chronic history", "reason": "Excludes 'acute'"}
}
"""
        },
        {
            "user": """
Source Concept: "cholera"
Source Context: Profuse watery diarrhea, dehydration, recent travel.
Candidate List:
HPO:6000904 | Positive Vibrio cholerae stool culture
HPO:0002014 | Diarrhea
HPO:0020106 | Severe giardiasis
""",
            "assistant": """
{
  "best_id": "NO_MATCH",
  "best_label": "No suitable match",
  "confidence": 0.0,
  "rationale": "Candidates are phenotypes or lab findings; source is an infectious disease.",
  "support": {"evidence": "No OMIM/ORPHA disease candidate", "reason": "No suitable target concept"}
}
"""
        }
    ],
    "user": """
Source Concept:
{{ source_concept }}

Source Context:
{{ source_context }}

Candidate List:
{{ candidate_list }}

Return STRICT JSON only.
If no candidate fits, use best_id="NO_MATCH", best_label="No suitable match", confidence=0.0.
Hard limits:
- Do not echo the source, context, or full candidate list.
- Do not include any fields not in the schema.
- If multiple options are close, pick one and explain briefly; do not list alternatives.
"""
}


PATIENT_NER_PROMPT = {
  "system": """
You are an expert clinical NER annotator.
Your task is to label disorder/problem mentions in clinical text using ONLY the ICD Chapter labels provided.

INPUTS:
- ICD_CHAPTERS: list of allowable labels (from Neo4j). Examples:
  "Endocrine, nutritional and metabolic diseases",
  "Diseases of the respiratory system",
  "Diseases of the musculoskeletal system and connective tissue",
  "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified", etc.

- CONCAT_TEXT: concatenation of Encounter.reasonCode + ChiefComplaint + Condition (in that order, separated by " | ").
- NARRATIVE_TEXT: the encounter narrative.

OUTPUT FORMAT (STRICT JSON):
{
  "patient_id": "str",
  "encounter_id": "str",
  "entities": [
    {
      "source": "concat" | "narrative",
      "start": 0,
      "end": 0,
      "text": "str",
      "label": "ICD Chapter EXACTLY as given in ICD_CHAPTERS",
      "assertion": "present" | "negated" | "uncertain",
      "temporality": "acute" | "chronic" | "recurrent" | "history" | "unspecified",
      "rationale": "str (≤1 sentence)"
    }
  ]
}

ANNOTATION RULES:
- Match by meaning, not wording.
- Use contiguous spans found verbatim in the text (0-based [start, end) indices).
- Choose the most specific phrase that represents a clinical condition, symptom, or disorder.
- Mark negated or uncertain mentions using 'assertion'.
- Determine temporality when clear; otherwise, 'unspecified'.
- Use ICD Chapters **exactly as provided**, no abbreviation or modification.
- Do not annotate medications, tests, vitals, or administrative details.
- If no entities apply, return "entities": [].
- Keep rationale concise (≤1 sentence).
""",

  "examples": [
    {
      "user": """
ICD_CHAPTERS:
- Endocrine, nutritional and metabolic diseases
- Diseases of the respiratory system
- Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified

patient_id: P010
encounter_id: E021

CONCAT_TEXT:
Excessive thirst and fatigue | Constant tiredness and dry mouth | Type 2 diabetes mellitus

NARRATIVE_TEXT:
The patient reported several months of increased thirst and urination, accompanied by fatigue. Laboratory testing revealed elevated fasting glucose, consistent with type 2 diabetes mellitus.
""",
      "assistant": """
{
  "patient_id": "P010",
  "encounter_id": "E021",
  "entities": [
    {
      "source": "concat",
      "start": 70,
      "end": 94,
      "text": "Type 2 diabetes mellitus",
      "label": "Endocrine, nutritional and metabolic diseases",
      "assertion": "present",
      "temporality": "chronic",
      "rationale": "Definitive diagnosis of an endocrine disorder."
    },
    {
      "source": "concat",
      "start": 0,
      "end": 29,
      "text": "Excessive thirst and fatigue",
      "label": "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified",
      "assertion": "present",
      "temporality": "chronic",
      "rationale": "Symptom-level description without diagnosis."
    }
  ]
}
"""
    },
    {
      "user": """
ICD_CHAPTERS:
- Diseases of the respiratory system
- Certain infectious and parasitic diseases
- Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified

patient_id: P012
encounter_id: E033

CONCAT_TEXT:
Persistent cough and fever | Productive cough with green sputum | Community-acquired pneumonia

NARRATIVE_TEXT:
She presented with a week of fever, chills, and productive cough. Chest X-ray confirmed right lower lobe consolidation, consistent with community-acquired pneumonia.
""",
      "assistant": """
{
  "patient_id": "P012",
  "encounter_id": "E033",
  "entities": [
    {
      "source": "concat",
      "start": 74,
      "end": 102,
      "text": "Community-acquired pneumonia",
      "label": "Diseases of the respiratory system",
      "assertion": "present",
      "temporality": "acute",
      "rationale": "Respiratory infection confirmed radiographically."
    },
    {
      "source": "narrative",
      "start": 29,
      "end": 43,
      "text": "productive cough",
      "label": "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified",
      "assertion": "present",
      "temporality": "acute",
      "rationale": "Symptom describing the illness presentation."
    }
  ]
}
"""
    },
    {
      "user": """
ICD_CHAPTERS:
- Diseases of the musculoskeletal system and connective tissue
- Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified

patient_id: P014
encounter_id: E045

CONCAT_TEXT:
Knee pain and stiffness | Aching knees when climbing stairs | Osteoarthritis of the knee

NARRATIVE_TEXT:
He described gradual worsening of knee discomfort over the past year, especially when walking or standing for long periods. Imaging confirmed degenerative osteoarthritis.
""",
      "assistant": """
{
  "patient_id": "P014",
  "encounter_id": "E045",
  "entities": [
    {
      "source": "concat",
      "start": 66,
      "end": 91,
      "text": "Osteoarthritis of the knee",
      "label": "Diseases of the musculoskeletal system and connective tissue",
      "assertion": "present",
      "temporality": "chronic",
      "rationale": "Degenerative joint disease confirmed radiographically."
    },
    {
      "source": "concat",
      "start": 0,
      "end": 20,
      "text": "Knee pain and stiffness",
      "label": "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified",
      "assertion": "present",
      "temporality": "chronic",
      "rationale": "Symptomatic presentation of musculoskeletal disorder."
    }
  ]
}
"""
    },
    {
      "user": """
ICD_CHAPTERS:
- Mental and behavioural disorders
- Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified

patient_id: P016
encounter_id: E057

CONCAT_TEXT:
Low mood and poor concentration | Feeling down and fatigued | Mild depressive episode

NARRATIVE_TEXT:
Patient reported persistent low mood and reduced motivation for several months. Denied suicidal thoughts. Diagnosed with mild depressive episode.
""",
      "assistant": """
{
  "patient_id": "P016",
  "encounter_id": "E057",
  "entities": [
    {
      "source": "concat",
      "start": 61,
      "end": 82,
      "text": "Mild depressive episode",
      "label": "Mental and behavioural disorders",
      "assertion": "present",
      "temporality": "chronic",
      "rationale": "Explicit psychiatric diagnosis per criteria."
    },
    {
      "source": "narrative",
      "start": 8,
      "end": 16,
      "text": "low mood",
      "label": "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified",
      "assertion": "present",
      "temporality": "chronic",
      "rationale": "Symptom description rather than disorder."
    },
    {
      "source": "narrative",
      "start": 72,
      "end": 95,
      "text": "Denied suicidal thoughts",
      "label": "Mental and behavioural disorders",
      "assertion": "negated",
      "temporality": "unspecified",
      "rationale": "Negated symptom within psychiatric context."
    }
  ]
}
"""
    },
    {
      "user": """
ICD_CHAPTERS:
- Diseases of the skin and subcutaneous tissue
- Certain infectious and parasitic diseases
- Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified

patient_id: P018
encounter_id: E063

CONCAT_TEXT:
Painful red rash on arm | Burning sensation and swelling | Cellulitis of upper limb

NARRATIVE_TEXT:
A rapidly spreading, tender rash was noted on the right forearm, with mild swelling and warmth. The patient was afebrile. Clinical impression: cellulitis of the upper limb.
""",
      "assistant": """
{
  "patient_id": "P018",
  "encounter_id": "E063",
  "entities": [
    {
      "source": "concat",
      "start": 67,
      "end": 89,
      "text": "Cellulitis of upper limb",
      "label": "Certain infectious and parasitic diseases",
      "assertion": "present",
      "temporality": "acute",
      "rationale": "Bacterial skin infection under infectious diseases."
    },
    {
      "source": "concat",
      "start": 0,
      "end": 24,
      "text": "Painful red rash on arm",
      "label": "Diseases of the skin and subcutaneous tissue",
      "assertion": "present",
      "temporality": "acute",
      "rationale": "Primary skin manifestation of infection."
    }
  ]
}
"""
    }
  ],

  "user": """
ICD_CHAPTERS:
{{ icd_chapters }}

patient_id: {{ patient_id }}
encounter_id: {{ encounter_id }}

CONCAT_TEXT:
{{ concat_text }}

NARRATIVE_TEXT:
{{ narrative_text }}
"""
}


PATIENT_NED_PROMPT = {
  "system": """
You are an expert clinical entity linker for ICD codes.

TASK
Given:
1) a single extracted mention with metadata,
2) a ranked list of ICD candidate codes (with scores and labels), and
3) other extracted mentions from the same note (for collective disambiguation),
choose the SINGLE best ICD code for the mention or abstain if none fits.

OUTPUT
Return STRICT JSON that preserves ALL original mention fields and appends:
- "icd_id": string | null               // e.g., "F32.0"; null if abstaining
- "icd_label": string | null            // the official title for icd_id; null if abstaining
- "confidence": number                  // 0–1 calibrated confidence for your final choice
- "linking_rationale": string           // ≤1 sentence, why this code matches the mention

IMPORTANT RULES
- Match the mention’s MEANING, not just wording; respect negation/temporality signals.
- Prefer the most specific candidate that exactly fits the mention text span and context.
- If the mention is clearly a symptom/sign (and no disorder-level diagnosis is stated),
  prefer an R-chapter (symptoms) code over mood/disorder diagnoses, unless the mention
  itself names a disorder (e.g., “major depressive episode”).
- Use collective disambiguation: prefer candidates consistent with other mentions
  (e.g., “fever”, “cough”, “consolidation” collectively support pneumonia).
- Respect assertion:
  • If the mention is negated (“assertion”: "negated"), abstain unless ICD coding
    guidelines in your setting require coding negated conditions (default: abstain).
  • If "uncertain", choose only if the ICD candidate reasonably covers suspected
    conditions; otherwise abstain.
- Do NOT invent codes not present in the candidate list.
- Break ties using (in order):
  1) semantic fit to the exact span and its label,
  2) clinical coherence with OTHER_MENTIONS,
  3) candidate score,
  4) greater specificity.
- When no candidate is a good fit, return icd_id=null and state why in the rationale.
- Keep the rationale concise (≤1 sentence).

VALIDATION
- Ensure "icd_id" and "icd_label" are consistent (label must correspond to the id from candidates).
- Confidence ∈ [0,1] and reflects your certainty in the final choice (not just the top candidate score).

""",

  "examples": [
    {
      "user": """
MENTION:
{
  "source": "narrative",
  "start": 8,
  "end": 16,
  "text": "low mood",
  "label": "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified",
  "assertion": "present",
  "temporality": "chronic",
  "rationale": "Symptom description rather than disorder."
}

CANDIDATES:
[
  {"score": 0.759, "label": "Unhappiness", "id": "R45.2"},
  {"score": 0.753, "label": "Depressive episode", "id": "F32"},
  {"score": 0.744, "label": "Mild depressive episode", "id": "F32.0"},
  {"score": 0.734, "label": "Unspecified mood [affective] disorder", "id": "F39"},
  {"score": 0.734, "label": "Persistent mood [affective] disorders", "id": "F34"},
  {"score": 0.732, "label": "Demoralization and apathy", "id": "R45.3"},
  {"score": 0.729, "label": "Depressive conduct disorder", "id": "F92.0"},
  {"score": 0.723, "label": "Organic mood [affective] disorders", "id": "F06.3"},
  {"score": 0.722, "label": "Persistent mood [affective] disorder, unspecified", "id": "F34.9"},
  {"score": 0.717, "label": "Moderate depressive episode", "id": "F32.1"}
]

OTHER_MENTIONS:
[
  {"text": "low mood", "label": "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified"}
]
""",
      "assistant": """
{
  "source": "narrative",
  "start": 8,
  "end": 16,
  "text": "low mood",
  "label": "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified",
  "assertion": "present",
  "temporality": "chronic",
  "rationale": "Symptom description rather than disorder.",
  "icd_id": "R45.2",
  "icd_label": "Unhappiness",
  "confidence": 0.83,
  "linking_rationale": "The span denotes a symptom; R45.2 ‘Unhappiness’ best matches and is consistent with the symptom chapter label."
}
"""
    },
    {
      "user": """
MENTION:
{
  "source": "concat",
  "start": 61,
  "end": 82,
  "text": "Mild depressive episode",
  "label": "Mental and behavioural disorders",
  "assertion": "present",
  "temporality": "chronic",
  "rationale": "Explicit psychiatric diagnosis per criteria."
}

CANDIDATES:
[
  {"score": 0.88, "label": "Mild depressive episode", "id": "F32.0"},
  {"score": 0.80, "label": "Depressive episode", "id": "F32"},
  {"score": 0.76, "label": "Moderate depressive episode", "id": "F32.1"}
]

OTHER_MENTIONS:
[
  {"text": "low mood", "label": "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified"}
]
""",
      "assistant": """
{
  "source": "concat",
  "start": 61,
  "end": 82,
  "text": "Mild depressive episode",
  "label": "Mental and behavioural disorders",
  "assertion": "present",
  "temporality": "chronic",
  "rationale": "Explicit psychiatric diagnosis per criteria.",
  "icd_id": "F32.0",
  "icd_label": "Mild depressive episode",
  "confidence": 0.92,
  "linking_rationale": "Exact textual and diagnostic match; most specific candidate consistent with context."
}
"""
    },
    {
      "user": """
MENTION:
{
  "source": "narrative",
  "start": 120,
  "end": 131,
  "text": "no pneumonia",
  "label": "Diseases of the respiratory system",
  "assertion": "negated",
  "temporality": "unspecified",
  "rationale": "Explicitly ruled out."
}

CANDIDATES:
[
  {"score": 0.74, "label": "Pneumonia, unspecified organism", "id": "J18.9"}
]

OTHER_MENTIONS:
[
  {"text": "fever", "label": "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified"}
]
""",
      "assistant": """
{
  "source": "narrative",
  "start": 120,
  "end": 131,
  "text": "no pneumonia",
  "label": "Diseases of the respiratory system",
  "assertion": "negated",
  "temporality": "unspecified",
  "rationale": "Explicitly ruled out.",
  "icd_id": null,
  "icd_label": null,
  "confidence": 0.98,
  "linking_rationale": "The mention is negated; abstaining from assigning a code."
}
"""
    }
  ],

  "user": """
MENTION:
{{ mention_json }}

CANDIDATES:
{{ candidates_json }}   // array of {"score": number, "label": "str", "id": "ICD code"}

OTHER_MENTIONS:
{{ other_mentions_json }}
"""
}
