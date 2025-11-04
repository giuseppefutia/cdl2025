PROMPT = """
# 🔧 Generation-Ready Prompt: FHIR-Style Encounters with Rich Narratives for NED (LLM-optimized, expanded symptom set)

## Objective
Generate **10 synthetic medical encounter records** for a **now 70-year-old patient**, covering a **10-year period** (from age 60 to 70).
The patient’s symptoms **wax and wane** over time — sometimes disappearing, sometimes returning — with **arrhythmias**, **transient global amnesia (TGA)**, and additional
**systemic or behavioral symptoms** such as **fever, headache, fatigue, muscle pain (myalgia), joint pain (arthralgia), nausea, insomnia, or aggressive behavior**.

## Global Rules (strict)
- Produce **synthetic, realistic, chronologically consistent** data. **Never** include real PHI.
- Write **dates in ISO 8601** and distribute across 10 distinct years.
- **Narratives are the priority**: rich, varied, natural language; avoid ICD/SNOMED/CPT-like phrasing.
- **Vary lexical choices intentionally** for similar entities (see Synonym Hints).
- Ensure **alignment** between the structured data and the narrative context.
- Alternate **CourseTrend** (improved ↔ worsened ↔ stable) over time.
- Include at least **3 encounters with arrhythmia** and **2 with TGA**; the remaining may include systemic or behavioral symptoms.
- Reflect realistic clinical evolution: symptom clusters may overlap or change across the years.

## Randomness & Diversity Controls
- Randomize but keep plausible: encounter dates, vitals, labs, and meds.
- Rotate **narrative syntax** and **sentence order** to increase lexical diversity.
- Use **natural, human phrasing** for symptoms (not medical textbook terms).

## Output Format (exact)
1) A **single Markdown table** with **exactly 10 rows** and **these 16 columns**:

| Encounter.period.start | Encounter.class | Encounter.reasonCode | ChiefComplaint | Condition | Comorbidities | Observation[vitals] | Observation[key] | DiagnosticReport | Procedure | MedicationStatement | Encounter.diagnosis.rank | Encounter.hospitalization.dischargeDisposition | Plan/FollowUp | CourseTrend | Notes |
|-------------------------|----------------|----------------------|----------------|-----------|----------------|---------------------|------------------|------------------|-----------|--------------------|--------------------------|-----------------------------------------------|----------------|-------------|-------|

2) Under each table row, write a **narrative paragraph (3–5 sentences, 60–100 words)** of natural clinical prose.

## Column Guidance
- Encounter.period.start: ISO date, spaced ~1 year apart.
- Encounter.class: Ambulatory | Emergency | Observation.
- Encounter.reasonCode: Plain-language reason (e.g., “memory lapse and pounding heartbeat”).
- ChiefComplaint: ≤12 words, natural phrasing.
- Condition: Primary issue (no coding terms).
- Comorbidities: Include arrhythmia subtype(s), TGA, or chronic/systemic conditions.
- Observation[vitals]: BP, HR, RR, Temp, SpO₂ — plausible values.
- Observation[key]: Disease-relevant measure(s) with units.
- DiagnosticReport: Key test(s) + result summary.
- Procedure: ECG, MRI/CT, labs, etc.
- MedicationStatement: Generic name, dose, route, frequency.
- Encounter.diagnosis.rank: 1–3.
- Encounter.hospitalization.dischargeDisposition: Home | Observation | Admitted.
- Plan/FollowUp: Short natural plan.
- CourseTrend: Improved | Worsened | Stable.
- Notes: ≤20 words.

## Narrative Requirements (critical)
- Sound like a **brief clinician note or discharge summary**.
- Include **temporal context**, **patient phrasing**, and **clinician reasoning**.
- Use **diverse wordings** for similar phenomena.
- Mention relevant tests, impressions, and management changes naturally.
- Symptoms can cluster — e.g., fever + myalgia; insomnia + irritability; arrhythmia + fatigue.

### Synonym Hints (use variety; examples, not limits)
- **Arrhythmia:** palpitations | fluttering | irregular thumping | racing heartbeat | skipped beats | erratic pulse  
- **TGA:** memory gap | lost track of time | blank spell | unable to recall | repetitive questioning  
- **Fever:** mild fever | low-grade temperature | felt flushed | burning up | warmth and chills  
- **Headache:** pounding head | pressure behind eyes | throbbing temples | dull ache  
- **Fatigue:** drained | low energy | exhausted | sluggish | run down  
- **Myalgia:** body aches | sore muscles | aching limbs | stiffness  
- **Arthralgia:** joint pain | sore knees | stiffness in hands | aching joints  
- **Nausea:** queasy | upset stomach | waves of nausea | felt sick  
- **Insomnia:** trouble sleeping | restless nights | couldn’t stay asleep | tossing and turning  
- **Aggressive behavior:** irritability | anger bursts | impatience | short temper | outbursts  

## Clinical Logic
- Ages progress 60 → 70; dates strictly ascending.
- Include ≥1 ECG report and ≥1 brain MRI/CT.
- Include ≥2 medication changes.
- Keep vitals and labs plausible; avoid contradictions (e.g., normal HR during AF episode).

## Hard Constraints
- Exactly **10 rows** + **10 narratives**.
- Narratives: **3–5 sentences each**, **~600–900 words total**.
- No codes, identifiers, or meta text.
- Self-contained and ready for **NER/NED** or **linguistic analysis**.

## Example (abbreviated)
| 2017-08-14 | Emergency | Sudden confusion episode | Couldn’t recall events for 3 hours | Transient memory loss | Hypertension, arrhythmia | 138/84 mmHg, HR 78 bpm, RR 16, Temp 36.9°C, SpO₂ 98% | Glucose 102 mg/dL (normal) | MRI brain: no acute lesion | Brain MRI | Atenolol 50 mg daily | 1 | Home | Neurology follow-up | Improved | First memory gap episode |
Narrative: Brought in by spouse after several hours of disorientation and repetitive questioning. Recognized family but couldn’t remember recent events. Neuro exam normal; MRI ruled out stroke. Memory returned by next morning. Discharged with neurology follow-up.

## Final Instruction
Generate the full table and narratives now, following all rules above.
"""