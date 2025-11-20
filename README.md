# cdl2025

`cdl2025` provides tools and pipelines for importing clinical ontologies
(ICD10, HPO, UMLS) into a Neo4j graph database, computing ontology
embeddings, and integrating patient-generated data for downstream
analytics.

Slides of the masterclass: [Google Drive](https://docs.google.com/presentation/d/1SCPnt3lXVT2C6ZjqYMRbfCXfkK5NEPQ4/edit?usp=sharing&ouid=112517546961694752607&rtpof=true&sd=true)

------------------------------------------------------------------------

## Table of Contents

-   [Overview](#overview)
-   [Installation](#installation)
-   [Data Pipeline](#data-pipeline)
    -   [ICD10 Import](#icd10-import)
    -   [HPO Import](#hpo-import)
    -   [UMLS Concept Mapping](#umls-concept-mapping)
    -   [Patient Data](#patient-data)
-   [Import Commands](#import-commands)
-   [Neo4j & APOC Virtualization](#neo4j--apoc-virtualization)
-   [Contributing](#contributing)
-   [License](#license)

------------------------------------------------------------------------

## Overview

This project supports:

-   Loading ICD10 codes, chapters, groups, and embeddings
-   Loading HPO ontology and embeddings
-   Mapping UMLS concepts to ICD10 and HPO
-   Ingesting patient-generated annotations
-   Creating virtualized views in Neo4j via APOC Data Virtualization

------------------------------------------------------------------------

## Installation

Create and activate a Python virtual environment:

``` bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.lock
```

------------------------------------------------------------------------

## Data Pipeline

### ICD10 Import

The ICD10 pipeline loads:

-   Disease codes + hierarchy\
-   Chapters\
-   Groups\
-   Embeddings

### HPO Import

The HPO pipeline loads:

-   The full HPO ontology\
-   Precomputed HPO embeddings

### UMLS Concept Mapping

Maps UMLS concepts (from `MRCONSO.RRF`) to:

-   ICD10 Codes\
-   HPO Terms

### Patient Data

Import patient-generated annotations (CSV-format).\
**TODO:** Add schema specification and examples.

------------------------------------------------------------------------

## Import Commands

### ICD10 Codes

``` bash
python -m factory.icd10 --backend neo4j --file icd102019syst_codes.txt
```

### ICD10 Chapters

``` bash
python -m factory.icd10_chapter --backend neo4j --file icd102019syst_chapters.txt
```

### ICD10 Groups

``` bash
python -m factory.icd10_group --backend neo4j --file icd102019syst_groups.txt
```

### ICD10 Embeddings

``` bash
python -m factory.icd10_embedding --backend neo4j
```

------------------------------------------------------------------------

### HPO Ontology

``` bash
python -m factory.hpo --backend neo4j
```

### HPO Embeddings

``` bash
python -m factory.hpo_embedding --backend neo4j
```

------------------------------------------------------------------------

### UMLS Mapping

``` bash
python -m factory.umls_map --backend neo4j --file MRCONSO.RRF
```

------------------------------------------------------------------------

### Patient Annotation Import

``` bash
python -m factory.patient_annotation --backend neo4j --file patient03.csv
```

------------------------------------------------------------------------

# Neo4j & APOC Virtualization

### 1. Download APOC

Download the appropriate release:

<https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/5.26.0>

Place it in Neo4j's `plugins/` directory.

------------------------------------------------------------------------

### 2. Update `neo4j.conf`

    dbms.security.procedures.unrestricted=apoc.*
    dbms.security.procedures.allowlist=apoc.*

------------------------------------------------------------------------

### 3. Update `apoc.conf`

    apoc.import.file.enabled=true
    apoc.import.file.use_neo4j_config=true

------------------------------------------------------------------------

### 4. Example: Install a Virtual CSV Source

``` cypher
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
```

------------------------------------------------------------------------

## Contributing

Pull requests are welcome!\
Please use feature branches and follow existing module patterns.

------------------------------------------------------------------------
