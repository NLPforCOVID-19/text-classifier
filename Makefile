DATA_DIR = data/CoronavirusTranslationSample20200326
OUTPUT_DIR = data

all: $(OUTPUT_DIR)/output.json

$(OUTPUT_DIR)/output.json: $(OUTPUT_DIR)/metadata.json
	python classifier.py -d $(DATA_DIR) $< $@

$(OUTPUT_DIR)/metadata.json: $(OUTPUT_DIR)/target_files.txt
	python metadata.py -d $(DATA_DIR) $< $@

$(OUTPUT_DIR)/target_files.txt:
	(cd $(DATA_DIR); find . -name "*.url") > $@

