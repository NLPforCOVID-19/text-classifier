DATA_DIR = data/CoronavirusTranslationData20200410a
OUTPUT_DIR = data
TO_BE_UPLOAD_DIR = data/tbupload

all: $(OUTPUT_DIR)/output.json
cleanall:
	@echo "Cleaning up..."
	rm $(OUTPUT_DIR)/output.json
	rm $(OUTPUT_DIR)/metadata.json
	rm $(OUTPUT_DIR)/target_files.txt
cleanfinal:
	@echo "Cleaning only output.json up..."
	rm $(OUTPUT_DIR)/output.json
cleanupload:
	rm $(OUTPUT_DIR)/uploadlist
	rm $(OUTPUT_DIR)/tbupload/*
upload:$(OUTPUT_DIR)/uploadlist $(OUTPUT_DIR)/output.json
	python uploader.py
	make cleanupload

$(OUTPUT_DIR)/output.json: $(OUTPUT_DIR)/metadata.json keywords.txt
	python classifier.py -d $(DATA_DIR) $< keywords.txt $@

$(OUTPUT_DIR)/metadata.json: $(OUTPUT_DIR)/target_files.txt
	python metadata.py -d $(DATA_DIR) $< $@

$(OUTPUT_DIR)/uploadlist:
	split -l 50 $(OUTPUT_DIR)/output.json $(TO_BE_UPLOAD_DIR)/output
	(cd $(OUTPUT_DIR)/tbupload && find .  -name "output*" ) > $@

$(OUTPUT_DIR)/target_files.txt:
	(cd $(DATA_DIR) && find . -name "*.url") > $@

