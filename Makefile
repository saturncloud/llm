
.PHONY: conda-update
conda-update:
	mamba env create -n bert-qa --file environment.yaml --force
	mamba env update -n bert-qa --file environment.test.yaml
