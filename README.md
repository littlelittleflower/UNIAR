# UNIAR--FEKG Inductive Reasoning
Inductive–Abductive Unseen-Node Reasoning on Fuzzy Event Graphs via LLM Calibration

## Dataset
The dataset can be downloaded from:
ACE_2005: https://catalog.ldc.upenn.edu/LDC2006T06
EventKG: https://eventkg.l3s.uni-hannover.de/data.html

process it to the directory ./data.

##  Requirements
To install the various python dependences

python 3.6

transformers 4.41.1

cuda 11.0

pytorch 1.7.1

dgl 0.6.1

torch_geometric 2.0.3

## LLM Prompt
```
python ./llm_calibration/llm_relgraph.py

```

## Meta-training & Fine-tuning
```
python main.py

```