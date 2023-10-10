from dataclasses import dataclass
from llm.prompt import Prompt


@dataclass
class TextSummarization(Prompt):
    system_message: str = "Create a concise summary of the given text."
    input_template: str = "Text: {text}"
    response_template: str = "Summary: {text}"
