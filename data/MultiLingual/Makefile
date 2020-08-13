Output_Directory=0804
Extract:
	python3 Goose-Extractor.py --input ./crowdsourcing20200615.jsonl --output ./annotation.jsonl
Partition:
	rm  -rf $(Output_Directory)/*
	python3 splitdata.py --input ./annotation.jsonl --output $(Output_Directory)

all: Extract Partition