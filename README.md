# BERT Simple Classifier 

## Environment

- To run this code, pytorch 1.3.0 and transformer 2.5.0 is recommended

## Preparing Model

- Put pretrained BERT under `dependencies/bert/<Model of your choice>`

## Preparing data

- Put `CoronavirusTranslationData20200421` and `crowdsourcing20200420.jsonl` in data folder
- Run preprocess.py on root directory which will produce `crowdsourcing20200420.processed.jsonl`
- Run Makefile on data folder with `CoronavirusTranslationData20200421` and `crowdsourcing20200420.processed.jsonl` in place, which will produce data necessary under data/0421

## Run BERT model

- run `Make BERT` on root directory
