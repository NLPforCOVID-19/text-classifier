BERT_MODEL_PATH=dependencies/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers
EPOCH=100
BERT_CE:
	python main.py --bert_path $(BERT_MODEL_PATH)  --use_ce --epochs $(EPOCH)
BERT_F1:
	python main.py --bert_path $(BERT_MODEL_PATH) --epochs $(EPOCH)
BERT_MD:
	python main.py --bert_path "bert-base-multilingual-cased" --expname "decoder_per_lang" --use_ce --epochs $(EPOCH)
BERT_MD_FT:
	python main.py --bert_path "bert-base-multilingual-cased" --expname "decoder_per_lang_finetuning" --finetuning --use_ce --epochs $(EPOCH)
BERT_MD_CHUNK_FT:
	python main.py --bert_path "bert-base-multilingual-cased" --expname "rnn_decoder_per_lang_train_by_chunk_finetuning" --train_by_chunk --use_ce --finetuning --epochs $(EPOCH) --chunk_len_limit 96 --context_size 24
BERT_MD_RNN_XR:
	python main.py --bert_path "bert-base-multilingual-cased" --expname "rnn_mixer_decoder_per_lang_nonfinetuning_combine_xr" --bertrnn --use_ce --epochs 200 --combine_xr
BERT_MD_RNN_CHUNK_FT:
	python main.py --bert_path "bert-base-multilingual-cased" --expname "rnn_decoder_per_lang_train_by_chunk_finetuning" --train_by_chunk --use_ce --combine_xr --bertrnn --finetuning --epochs $(EPOCH) --chunk_len_limit 96 --context_size 24

#cl-tohoku/bert-base-japanese
BERT_MD_RNN:
	python main.py --bert_path "bert-base-multilingual-cased" --expname "rnn_mixer_decoder_per_lang" --bertrnn --finetuning --use_ce --epochs $(EPOCH)  --sample_size 32 --sample_count 8
BERT_MD_RNN_NONFT:
	python main.py --bert_path "bert-base-multilingual-cased" --expname "rnn_mixer_decoder_per_lang_nonfinetuning" --bertrnn --use_ce --epochs 200
BERT_SD:
	python main.py --bert_path $(BERT_MODEL_PATH) --use_ce --expname "decoder_for_all_lang" --epochs $(EPOCH) --decoder_sharing
BERT_MD_AT:
	python main.py --bert_path "bert-base-multilingual-cased" --expname "decoder_per_lang" --article_level --use_ce --finetuning --epochs $(EPOCH) --article_len 256
BERT_MD_CHUNK_NONFT:
	python main.py --bert_path "bert-base-multilingual-cased" --expname "rnn_decoder_per_lang_train_by_chunk" --train_by_chunk --use_ce --bertrnn --epochs $(EPOCH) --chunk_len_limit 256
BERT_MD_CHUNK_NONFT_F1:
	python main.py --bert_path "bert-base-multilingual-cased" --expname "rnn_decoder_per_lang_train_by_chunk_f1" --train_by_chunk --use_f1 --bertrnn --epochs $(EPOCH) --chunk_len_limit 256