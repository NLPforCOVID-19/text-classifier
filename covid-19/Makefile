DATA_DIR = data/CoronavirusTranslationData20200331a
OUTPUT_DIR = data

all: $(OUTPUT_DIR)/output.json
clean: rm $(OUTPUT_DIR)/output.json

$(OUTPUT_DIR)/output.json: $(OUTPUT_DIR)/metadata.json
	python classifier.py -d $(DATA_DIR) $< $@

$(OUTPUT_DIR)/metadata.json: $(OUTPUT_DIR)/target_files.txt
	python metadata.py -d $(DATA_DIR) $< $@

$(OUTPUT_DIR)/target_files.txt:
	(cd $(DATA_DIR); find . -name "*.url") > $@

