# Saturn Cloud LLM (Un)Framework

The Saturn Cloud LLM Framework is a set of tools for several application level, 
as well as functional tasks around LLMs.

Application Tasks:
- RAG QA
- Text Summarization
- NER
- Automated Tagging

Functional Tasks:
- fine tuning
- batch inference
- model serving

This repository is designed to be a framework for common tasks one often performs with LLMs. 
This repository is also designed to be easily read, so that if your needs go beyond what is provided here, 
that it is easy for you to fork this repository, or build your own framework on top of your existing code.
You should fork this repository or build your own framework as soon as this repository stops making your life
easier.

## Structure of the repository

- llm module - this module contains all the "library" code to facilitate LLM applications, as well as LLM functional tasks.
- build_examples - this directory contains scripts used to prepare data used in examples. Users are not expected to use this directory
- starting_points - this directory contains code templates that you can implement in order to apply this repository to your own data.
- examples - this directory contains examples of using the framework on sample datasets. You can think of examples as the code in starting_points, implemented for specific datasets.

## Concepts

This repository uses a few concepts

- Model Config: We have a registry of common models supported by this framework. Model Configs include common parameters for each model, as well as the PromptFormat for the model 
- Prompt Format: is the format that was used to train the model. It is a good idea to use the Prompt Format for a given model, but sometimes not essential. 
  
    For example the Llama 2 chat model expects prompts to follow this style:
    
    """
    <s>[INST] <<SYS>>
    {system_message}
    <</SYS>>
    
    {input} [/INST] {response} </s>
    """
    
    Whereas Vicuna expects prompts to follow this style:
    
    """
    <s> {system_message}
    USER: {input}
    ASSISTANT: {response}
    </s>
    """
- Prompts: Prompts include specific system messages, examples (for few-shot learning) 
and some other formatting. Prompts can be mixed with different PromptFormats.

- Configuration. We've written scripts for tasks (such as fine tuning, batch inference, model serving)
So that you can ideally run these tasks without having to write any code at all. To do so we
rely on a lightweight yaml configuration to direct the specifics of each task. 
