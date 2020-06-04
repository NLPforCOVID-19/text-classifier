# Rules to train/run classifiers
# See `README.md` for details.

DATA_DIR = ../data
CLASSES_FILE = classes.txt  # list of classes

# Pretrained BERT
BERT_DIR = bert
BERT_MODEL = $(BERT_DIR)/Japanese_L-12_H-768_A-12_E-30_BPE_transformers
BERT_MODEL_URL = http://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JapaneseBertPretrainedModel/Japanese_L-12_H-768_A-12_E-30_BPE_transformers.zip

# keyword-matching classifier
test-kwclassifier: $(DATA_DIR)/annotations-test.jsonl keywords.txt
	python kwclassifier.py $(DATA_DIR)/annotations-test.jsonl keywords.txt $(DATA_DIR)/annotations-test-kwclassifier.jsonl
	python eval.py --gold $(DATA_DIR)/annotations-test.jsonl --pred $(DATA_DIR)/annotations-test-kwclassifier.jsonl --classes classes.txt
run-kwclassifier: $(DATA_DIR)/textdata.jsonl keywords.txt
	python kwclassifier.py $(DATA_DIR)/textdata.jsonl keywords.txt $(DATA_DIR)/textdata-kwclassifier.jsonl

# simple BERT classifier
$(BERT_MODEL):
	mkdir -p $(BERT_DIR)
	wget $(BERT_MODEL_URL) -P $(BERT_DIR)
	unzip $(BERT_MODEL).zip -d $(BERT_DIR)

BERTSIMPLE_MODEL = bertsimple/classifier.pth
train-bertsimple: $(DATA_DIR)/annotations-train.jsonl $(CLASSES_FILE) $(BERT_MODEL)
	mkdir -p $(dir $(BERTSIMPLE_MODEL))
	python bertsimple.py train --gpu --pos-weight --batch 64 --epoch 100 --bert-model $(BERT_MODEL) --text-file $< --model-path $(BERTSIMPLE_MODEL) --classes-file $(CLASSES_FILE)
test-bertsimple: $(DATA_DIR)/annotations-test.jsonl $(BERTSIMPLE_MODEL) $(CLASSES_FILE)
	python bertsimple.py test --gpu --batch 256 --bert-model $(BERT_MODEL) --text-file $< --model-path $(BERTSIMPLE_MODEL) --output-file $(DATA_DIR)/annotations-test-bertsimple.jsonl --classes-file $(CLASSES_FILE)
	python eval.py --gold $(DATA_DIR)/annotations-test.jsonl --pred $(DATA_DIR)/annotations-test-bertsimple.jsonl --classes classes.txt
run-bertsimple: $(DATA_DIR)/textdata.jsonl $(BERTSIMPLE_MODEL) $(CLASSES_FILE)
	python bertsimple.py test --gpu --batch 256 --bert-model $(BERT_MODEL) --text-file $< --model-path $(BERTSIMPLE_MODEL) --output-file $(DATA_DIR)/textdata-bertsimple.jsonl --classes-file $(CLASSES_FILE)

