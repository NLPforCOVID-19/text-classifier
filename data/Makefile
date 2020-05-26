DATA_DIR = ./CoronavirusTranslationData20200421
OUTPUT_DIR = .
TO_BE_UPLOAD_DIR = ./tbupload

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
	python3 uploader.py
	make cleanupload
splitdata:
	sed -n '1,2440 p' crowdsourcing20200420.processed.jsonl > 0421/crowdsourcing.train
	sed -n '2441,2748 p' crowdsourcing20200420.processed.jsonl > 0421/crowdsourcing.dev
	sed -n '2749,3050 p' crowdsourcing20200420.processed.jsonl > 0421/crowdsourcing.test

$(OUTPUT_DIR)/output.json: $(OUTPUT_DIR)/metadata.json keywords.txt
	python3 classifier.py -d $(DATA_DIR) $< keywords.txt $@

$(OUTPUT_DIR)/metadata.json: $(OUTPUT_DIR)/target_files.txt
	python3 metadata.py -d $(DATA_DIR) $< $@

$(OUTPUT_DIR)/uploadlist:
	split -l 50 $(OUTPUT_DIR)/output.json $(TO_BE_UPLOAD_DIR)/output
	(cd $(OUTPUT_DIR)/tbupload && find .  -name "output*" ) > $@

$(OUTPUT_DIR)/target_files.txt:
	(cd $(DATA_DIR) && find . -name "*.url") > $@

