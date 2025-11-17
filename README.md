# cdl2025

To install all the required Python libraries, you can create a virtual environment as follows:

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.lock
```

## Download
TODO

## Data
TODO

### Import ICD10 Code Dataset

Import ICD10 code diseases and their hierarchical connections:

```bash
python -m factory.icd10 --backend neo4j --file icd102019syst_codes.txt
```

Import ICD10 chapters and their connections with diseases:

```bash
python -m factory.icd10_chapter --backend neo4j --file icd102019syst_chapters.txt
```

Import ICD10 groups and their connections with chapters and diseases:

```bash
python -m factory.icd10_group --backend neo4j --file icd102019syst_groups.txt
```

Import ICD10 embeddings:

```bash
python -m factory.icd10_embedding --backend neo4j
```

### Import the HPO Ontology Dataset

Import HPO ontology:

```bash
python -m factory.hpo --backend neo4j
```

Import HPO embeddings:

```bash
python -m factory.hpo_embedding --backend neo4j
```

### Map UMLS concept to ICD10 and HPO Ontologies

```bash
python -m factory.umls_map --backend neo4j --file MRCONSO.RRF
```

### Patient-generated Data
TODO

## Import
TODO

```bash
python -m factory.patient_annotation --backend neo4j --file patient03.csv
```

# Virtualization

Download the release: https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/5.26.0

In the neo4j.conf
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*

In the apoc.conf file
apoc.import.file.enabled=true
apoc.import.file.use_neo4j_config=true

CALL apoc.dv.catalog.install(
  "patient", "cdl2025",
  {
    type: "CSV",
    url: "file:///patient.csv",
    labels: ["Patient"],
    query: "map.PatientID = $patientId",
    desc: "Patient details with patientId"
  }
);



## Utilities for Understanding
* https://chatgpt.com/c/6904bb2f-21d0-8332-913f-aae184bcf8f6
* https://chatgpt.com/c/6904beb2-04fc-8330-a624-5f6d51eab2b4
* https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct
* https://chatgpt.com/c/6904b843-21ac-8329-91dc-641d69f14427
* https://chatgpt.com/c/6904a4f1-3f80-8322-adc0-21bc0c481a43
* https://chatgpt.com/c/6904a093-1148-8323-9f49-881ece410edb
* https://github.com/JHnlp/BioCreative-V-CDR-Corpus (Useful for the evaluation)
* https://chatgpt.com/c/69048b84-dc0c-832a-abf6-e0d275dcbc4a
* AGENT: https://chatgpt.com/c/690ddcc1-af80-832a-8ea2-0de50c5114e6