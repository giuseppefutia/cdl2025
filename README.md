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

### Patient-generated Data
TODO

## Import
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

### Import the HPO Ontology Dataset

```bash
python -m factory.hpo --backend neo4j
```

### Create HPO Textual Embeddings

```bash
python -m factory.hpo_embedding --backend neo4j
```

### Map UMLS concept to ICD10 and HPO Ontologies

```bash
python -m factory.umls_map --backend neo4j --file MRCONSO.RRF
```

## Utilities for Understanding
* https://chatgpt.com/c/6904bb2f-21d0-8332-913f-aae184bcf8f6
* https://chatgpt.com/c/6904beb2-04fc-8330-a624-5f6d51eab2b4
* https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct
* https://chatgpt.com/c/6904b843-21ac-8329-91dc-641d69f14427
* https://chatgpt.com/c/6904a4f1-3f80-8322-adc0-21bc0c481a43
* https://chatgpt.com/c/6904a093-1148-8323-9f49-881ece410edb
* https://github.com/JHnlp/BioCreative-V-CDR-Corpus (Useful for the evaluation)
* https://chatgpt.com/c/69048b84-dc0c-832a-abf6-e0d275dcbc4a