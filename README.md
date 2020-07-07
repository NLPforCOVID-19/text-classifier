# BERT Simple Classifier 

## Environment

- To run this code, pytorch 1.3.0 and transformer 2.5.0 is recommended

## Preparing Model

- Put pretrained BERT under `dependencies/bert/<Model of your choice>`

## Preparing data

- Put `CoronavirusTranslationData20200421` and `crowdsourcing20200420.jsonl` in data folder

- Run `make all` on data folder containing `CoronavirusTranslationData20200421` and `crowdsourcing20200420.processed.jsonl`, which will produce `output.json`
- Run preprocess.py on root directory which will produce `crowdsourcing20200420.processed.jsonl`
- Run `make splitdata` on data folder to generate `data/0421` which is needed for run the model

## Run BERT model

- run `Make BERT` on root directory
