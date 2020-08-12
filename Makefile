BERT_MODEL_PATH=dependencies/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers
EPOCH=100
BERT_CE:
	CUDA_VISIBLE_DEVICES=1 python main.py --bert_path $(BERT_MODEL_PATH)  --use_ce --epochs $(EPOCH)
BERT_F1:
	CUDA_VISIBLE_DEVICES=1 python main.py --bert_path $(BERT_MODEL_PATH) --epochs $(EPOCH)
BERT_MD:
	CUDA_VISIBLE_DEVICES=0 python main.py --bert_path "bert-base-multilingual-cased" --expname "decoder_per_lang" --use_ce --finetuning --epochs $(EPOCH)
BERT_MD_RNN:
	CUDA_VISIBLE_DEVICES=3 python main.py --bert_path "bert-base-multilingual-cased" --expname "rnn_mixer_decoder_per_lang" --bertrnn --finetuning --use_ce --epochs $(EPOCH)  --sample_size 32 --sample_count 8
BERT_MD_RNN_NONFT:
	CUDA_VISIBLE_DEVICES=1 python main.py --bert_path "bert-base-multilingual-cased" --expname "rnn_mixer_decoder_per_lang_nonfinetuning" --bertrnn --use_ce --epochs $(EPOCH)
BERT_SD:
	CUDA_VISIBLE_DEVICES=1 python main.py --bert_path $(BERT_MODEL_PATH) --use_ce --expname "decoder_for_all_lang" --epochs $(EPOCH) --decoder_sharing