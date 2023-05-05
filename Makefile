
.PHONY: conda-update
conda-update:
	mamba env create -n bert-qa --file environment.yaml --force
	mamba env update -n bert-qa --file environment.test.yaml

.PHONY: api
api:
	python -m bert_qa.api
