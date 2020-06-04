DATA_DIR = data/CoronavirusTranslationData20200331a
OUTPUT_DIR = data

all: $(OUTPUT_DIR)/output.jsonl

$(OUTPUT_DIR)/output.jsonl: $(OUTPUT_DIR)/metadata.jsonl keywords.txt
	python classifier.py -d $(DATA_DIR) $< keywords.txt $@

$(OUTPUT_DIR)/metadata.jsonl: $(OUTPUT_DIR)/target_files.txt
	python metadata.py -d $(DATA_DIR) $< title_h1_domains.txt $@

$(OUTPUT_DIR)/target_files.txt:
	(cd $(DATA_DIR) && find . -name "*.url") > $@

