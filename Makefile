BERT_MODEL_PATH=dependencies/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers
EPOCH=100
BERT_CE:
	CUDA_VISIBLE_DEVICES=0 python main.py --bert_path $(BERT_MODEL_PATH) --use_ce --epochs $(EPOCH)
BERT_F1:
	CUDA_VISIBLE_DEVICES=1 python main.py --bert_path $(BERT_MODEL_PATH) --epochs $(EPOCH)