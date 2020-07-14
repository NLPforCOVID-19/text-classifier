# Makefile for data preprocessing
# Usage:
#   - Put `CoronavirusTranslationData20200427.zip` and `crowdsourcing20200511.jsonl.gz` in `data`
#   - Run `make data`
# Data files will be created in `data`
#   - metadata.jsonl: metadata of HTML/XML files
#   - textdata.jsonl: text data extracted from HTML files
#   - annotatinos-{train,test}.jsonl: text data with class labels annotated

DATA_DIR = data
#TRANSLATION_DATA = $(DATA_DIR)/CoronavirusTranslationData20200427
#ANNOTATION_DATA = $(DATA_DIR)/crowdsourcing20200511.jsonl
TRANSLATION_DATA = $(DATA_DIR)/covid19-data-20200618
ANNOTATION_DATA = $(DATA_DIR)/crowdsourcing20200615.jsonl

data: $(DATA_DIR)/annotations.jsonl $(DATA_DIR)/annotations-train.jsonl $(DATA_DIR)/annotations-test.jsonl

# Prepare source data
# $(TRANSLATION_DATA): $(TRANSLATION_DATA).zip
# 	unzip $< -d $(DATA_DIR)
# 	touch $(TRANSLATION_DATA)
$(TRANSLATION_DATA): $(TRANSLATION_DATA).tar.gz
	tar xvzf $< -C $(DATA_DIR)
	mv $(DATA_DIR)/covid19 $(TRANSLATION_DATA)
	touch $(TRANSLATION_DATA)
$(ANNOTATION_DATA): $(ANNOTATION_DATA).gz
	gzip -dc < $< > $@

# List of files to process
$(DATA_DIR)/target_files.txt: $(TRANSLATION_DATA)
	(cd $(TRANSLATION_DATA) && find . -name "*.url") > $@

# Metadata of HTML/XML files
$(DATA_DIR)/metadata.jsonl: $(DATA_DIR)/target_files.txt
	python preprocess/metadata.py -d $(TRANSLATION_DATA) $< $@
#	python preprocess/metadata.py -j 256 -d $(TRANSLATION_DATA) $< $@

# Text data extracted from HTML files
$(DATA_DIR)/textdata.jsonl: $(DATA_DIR)/metadata.jsonl
	python preprocess/extracttext.py -d $(TRANSLATION_DATA) $< $@
#	python preprocess/extracttext.py -j 256 -d $(TRANSLATION_DATA) $< $@

# Text data with class labels
$(DATA_DIR)/annotations.jsonl: $(DATA_DIR)/textdata.jsonl $(ANNOTATION_DATA)
	python preprocess/annotations.py $(DATA_DIR)/textdata.jsonl $(ANNOTATION_DATA) $@
$(DATA_DIR)/annotations-train.jsonl $(DATA_DIR)/annotations-test.jsonl: $(DATA_DIR)/annotations.jsonl
	python preprocess/datasplit.py --test-size 0.1 $< $(DATA_DIR)/annotations-train.jsonl $(DATA_DIR)/annotations-test.jsonl

