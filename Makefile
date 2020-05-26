BERT_MODEL_PATH=dependencies/bert/L-6_H-768_A-6_E-40_BPE
BERT:
	python main.py --bert_path $(BERT_MODEL_PATH)