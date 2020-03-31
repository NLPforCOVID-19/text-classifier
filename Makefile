DATA_DIR = data/CoronavirusTranslationSample20200326
OUTPUT_DIR = data

all: $(OUTPUT_DIR)/output.jsonl

$(OUTPUT_DIR)/output.jsonl: $(OUTPUT_DIR)/metadata.jsonl
	python classifier.py -d $(DATA_DIR) $< $@

$(OUTPUT_DIR)/metadata.jsonl: $(OUTPUT_DIR)/target_files.txt
	python metadata.py -d $(DATA_DIR) $< $@

$(OUTPUT_DIR)/target_files.txt:
	(cd $(DATA_DIR); find . -name "*.url") > $@

