## This is the official repository for the paper *OCMR Is All You Need for Chemical Structure Recognition*


### Usage

1. Pull image from dockerhub. This image contains the tools of OCMR/MolVec/OSRA/Imago.

```bash
**docker pull limingisafish/ocmr:v4**
```
2. Starting the docker container on local port 1234.

```bash
docker run --rm -ti -p 1234:5632 limingisafish/ocmr:v4
```
3.  Translate molecular image to SMILES.
```bash
python test.py

## {'imago': 'N#Cc1ccccc1C#N', 'molvec': 'N#CC1=C(C#N)C=CC=C1', 'osra': 'N#Cc1ccccc1C#N', 'ocmr': 'N#CC1=C(C#N)C=CC=C1'}
```

### Benchmark Test


#### Environment 

```docker
conda env create -f requirement.yml
conda activate ocmr_bench
```
#### Test PATENT dataset

1. Path to PATENT dataset
```
|----Data/
      |----Patent/
|----Evaluator_patent.py
```
2. run testing
```python
python Evaluator_patent.py
```
3 the result will be saved in `Patent_res.csv`.
```
OCMR acc：415/520
MolVec acc：318/520
OSRA acc：342/520
Imago acc：120/520
```


#### Test UOB，CLEF，USPTO，JPO benchmark

```
|----Data/
      |----public_data/
            |----UOB/
            |----MolVec/
            |----OSRA/
            |----Imago/
|----Evaluator_public.py
```
2. run testing
```python
python Evaluator_benchmarks.py
```
1. the result will be saved in `all_result_public.xlsx` and `public_res.csv`.
   
|  | SMI | Tanimoto |
| --- | --- | --- |
| Group |  |  |
| CLEF | 0.650202 | 0.931824 |
| JPO | 0.604444 | 0.879941 |
| UOB | 0.859930 | 0.980701 |
| USPTO | 0.746284 | 0.950304 |
