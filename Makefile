
.PHONY: conda-update
conda-update:
	mamba env create -n llm --file environment.yaml --force
	mamba env update -n llm --file environment.test.yaml


.PHONY: chat-cmdline
chat-cmdline:
	python llm/qa/cli/main.py chat cmdline --rephrase


.PHONY: chat-streamlit
chat-streamlit:
	python llm/qa/cli/main.py chat streamlit


.PHONY: format
format:
	black -l 100 .

.PHONY: test
test:
	pytest tests

