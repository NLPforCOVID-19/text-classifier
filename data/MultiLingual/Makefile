Output_Directory=./0804
DATA_DIR = ./covid19
DATASET_DIR = .
OUTPUT_DIR = .
Extract:
	python3 Goose-Extractor.py --input ./crowdsourcing20200615.jsonl --output ./annotation.jsonl
Partition:
	rm  -rf $(Output_Directory)/*
	python3 splitdata.py --input ./annotation.jsonl --input_retreated ./output.json --output $(Output_Directory)

all: Extract Partition

$(OUTPUT_DIR)/output.json: $(OUTPUT_DIR)/metadata.json keywords.txt
	python3 dataset_completion.py -d $(DATA_DIR) $< keywords.txt $@

$(OUTPUT_DIR)/metadata.json: $(OUTPUT_DIR)/target_files.txt
	python3 metadata.py -d $(DATA_DIR) $< $@


$(OUTPUT_DIR)/target_files.txt:
	(cd $(DATA_DIR) && find . -name "*.url") > $@

Get_Metadata: $(OUTPUT_DIR)/metadata.json
Get_Text: $(OUTPUT_DIR)/output.json
Get_Text_Ja: $(OUTPUT_DIR)/metadata.json keywords.txt
	python3 dataset_completion.py -d $(DATA_DIR) $< keywords.txt $@