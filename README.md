# BERT Simple Classifier 

## Environment

- To run this code, pytorch 1.3.0 and transformer 2.5.0 is recommended

## Preparing data

- Make sure you have `crowdsourcing<Date>.jsonl` in `data/Multilingual` folder
- Run `make all` 
    - `Extract` target will produce `annotation.json` which contains both annotation and crawled text
    - `Partition` target will partition data in `annotation.json` into train/dev/test sets randomly
    - To save time, you may run `Make Partition` using the existing `annotation.json`
## Run BERT model

- run `make <experiment target>` on root directory with the experiment configuration of your choice
