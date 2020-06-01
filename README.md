# Simple text classifier

- Put the following data files in `data` (or modify `TRANSLATION_DATA` and `ANNOTATION_DATA` in `Makefile`).
  - `CoronavirusTranslationData20200427.zip`
  - `crowdsourcing20200511.jsonl.gz`
- Run `make data` to build metadata and train/test datasets.  All data files are created under `data`.
- Train/run a classifier under `classifiers`.  See `classifiers/README.md` for details.

