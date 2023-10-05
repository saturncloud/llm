# Streamlit UI

An example interface for chatting with LLMs on different tasks.

## Run

The app supports multiple backends, depending on where you are running your model.

### Transformers Engine

Run models in a background process on the same machine as streamlit. This is the default backend if no arguments are given.
```
streamlit run examples/streamlit_ui/Chat.py
```

To run a different model, pass arguments into streamlit like so
```
streamlit run examples/streamlit_ui/Chat.py -- transformers --model-id <model-id>
```

### VLLM-Client Engine

Connect to a model running in a VLLM inference server. Even though the model is not running on the same machine,
you may still pass `model-id` in order to use a prompt format configured for the model (assuming it is a known model).

```
streamlit run examples/streamlit_ui/Chat.py -- vllm-client <url> --model-id <model-id>
```

## Tasks
### Chat:
Basic chat with a configurable system message, and message history retention.

### Document QA
Answer questions based on contexts retrieved from a document store

Dataset must be properly formatted ahead of time. The following environment variables configure which dataset is used for QA.

```
QA_DATASET_PATH: Path to the dataset
QA_INDEX_PATH: Path to the dataset's FAISS index built over embeddings (defaults: "<dataset>.faiss")
QA_CONTEXT_MODEL: Model ID used for embedding questions for context search (default: "sentence-transformers/multi-qa-mpnet-base-dot-v1")
```

### Text Summarization

Summarize important information from long passages of text.