
.PHONY: conda-update
conda-update:
	mamba env create -n bert-qa --file environment.yaml --force
	mamba env update -n bert-qa --file environment.test.yaml

.PHONY: streamlit-qa
streamlit-qa:
	streamlit run llm/qa/streamlit/app.py


.PHONY: chat-qa
chat-qa:
	python llm/qa/cli/main.py chat --rephrase
