from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Optional
if TYPE_CHECKING:
    from llm.model_configs import ModelConfig


@dataclass
class Role:
    prefix: str = ""
    suffix: str = ""

    def render(self, text: str, with_prefix: bool = True, with_suffix: bool = True) -> str:
        if with_prefix:
            text = f"{self.prefix}{text}"
        if with_suffix:
            text = f"{text}{self.suffix}"
        return text

    @property
    def stop_strings(self) -> List[str]:
        # Inlcude any suffix/prefix that is not all whitespace as a strop string
        return [s for s in (self.prefix, self.suffix) if s.strip()]


@dataclass
class PromptFormat:
    """
    Prompt roles and formatting options to control how messages between a user
    and an AI assistant are rendered.
    """
    input: Role = field(default_factory=Role)
    response: Role = field(default_factory=Role)
    system: Role = field(default_factory=Role)
    contexts: Role = field(default_factory=Role)
    examples: Role = field(default_factory=Role)

    # Nested roles are wrapped within the input role
    system_nested: bool = False
    contexts_nested: bool = True

    # Strip prefixes/suffixes before returning them as strop strings
    strip_stop_strings: bool = False
    # Separator between each message (instruction, response)
    message_separator: str = "\n"
    # Separator between instruction components (system, contexts, input)
    content_separator: str = "\n"
    # String added before each message input
    BOS: str = ""
    # String added after each message response
    EOS: str = ""

    def join_messages(self, texts: Iterable[str]) -> str:
        return self.message_separator.join(texts)

    def join_contents(self, contents: Iterable[str]) -> str:
        return self.content_separator.join(contents)

    @property
    def stop_strings(self) -> List[str]:
        stop_strings = (
            self.input.stop_strings
            + self.response.stop_strings
            + self.system.stop_strings
            + self.contexts.stop_strings
            + self.examples.stop_strings
        )
        if self.strip_stop_strings:
            return [s.strip() for s in stop_strings]
        return stop_strings


@dataclass
class UserAssistantFormat(PromptFormat):
    input: Role = field(default_factory=lambda: Role("User: "))
    response: Role = field(default_factory=lambda: Role("Assistant: "))


@dataclass
class RedpajamaFormat(PromptFormat):
    input: Role = field(default_factory=lambda: Role("<human>: "))
    response: Role = field(default_factory=lambda: Role("<bot>: "))


@dataclass
class VicunaFormat(PromptFormat):
    input: Role = field(default_factory=lambda: Role("USER: "))
    response: Role = field(default_factory=lambda: Role("ASSISTANT: "))
    BOS: str = "<s>"
    EOS: str = "</s>"


@dataclass
class Llama2Format(PromptFormat):
    input: Role = field(default_factory=lambda: Role(prefix="[INST] ", suffix=" [/INST]"))
    system: Role = field(default_factory=lambda: Role(prefix="<<SYS>>\n", suffix="\n<</SYS>>\n\n"))
    system_nested: bool = True
    message_separator: str = ""
    BOS: str = "<s>"
    EOS: str = "</s>"


@dataclass
class TogetherLlama2Format(PromptFormat):
    # Simplified version of Llama2 roles used in togethercomputer/Llama-2-7B-32K-Instruct
    # The exact formatting differs a bit from training data and model card, this is a mix of the two
    # that does well for inference.
    input: Role = field(default_factory=lambda: Role(prefix="[INST]  ", suffix="  [/INST]\n\n"))
    system_nested: bool = True
    message_separator: str = ""
    # Model tends to generate "[/INST]" with different spacings
    strip_stop_strings: bool = True


@dataclass
class ChatMLFormat(PromptFormat):
    input: Role = field(default_factory=lambda: Role(prefix="<|im_start|>user\n", suffix="<|im_end|>"))
    response: Role = field(default_factory=lambda: Role(prefix="<|im_start|>assistant\n", suffix="<|im_end|>"))
    system: Role = field(default_factory=lambda: Role(prefix="<|im_start|>system\n", suffix="<|im_end|>"))


@dataclass
class DollyFormat(PromptFormat):
    input: Role = field(default_factory=lambda: Role(prefix="\n### Instruction:\n"))
    response: Role = field(default_factory=lambda: Role(prefix="\n### Response:\n"))


@dataclass
class Message:
    input: str
    response: Optional[str] = None
    contexts: List[str] = field(default_factory=list)


@dataclass
class Prompt:
    system_message: str = ""
    examples: List[Message] = field(default_factory=list)
    format: PromptFormat = field(default_factory=PromptFormat)

    # Applies to message input string separate from contexts
    input_template: str = "{text}"
    # Applies to individual contexts. May also reference {index}
    context_template: str = "{text}"
    # Applies to previous responses, and appended with "" to prompt the next response
    response_template: str = "{text}"

    @classmethod
    def from_model_config(cls, model_config: ModelConfig, **kwargs):
        return cls(format=model_config.format, **kwargs)

    @property
    def stop_strings(self) -> List[str]:
        return self.format.stop_strings

    def render(
        self,
        messages: List[Message],
        with_system: bool = True,
        with_contexts: bool = True,
        with_examples: bool = True,
        strip: bool = True,
    ) -> str:
        """
        Render the prompt over a series of messages
        """
        results: List[str] = []

        # Format examples
        if with_examples and self.examples:
            examples_str = self.render_examples(with_system=with_system, with_contexts=with_contexts)
            results.append(examples_str)
            # System messed is included with examples
            with_system = False

        num_messages = len(messages)
        if num_messages == 0 and len(results) == 0:
            # No messages, and no examples. Return system message.
            if with_system and self.system_message:
                system_message = self.render_system()
                if self.format.system_nested:
                    system_message = self.format.input.render(system_message, with_suffix=False)
                return system_message
            return ""

        # Format conversation
        for i, message in enumerate(messages):
            last = (i == num_messages - 1)
            message_str = self.render_message(
                message,
                last=last,
                with_system=(with_system and (i == 0)),
                with_contexts=with_contexts,
            )
            results.append(message_str)

        final_str = self.format.join_messages(results)
        if strip:
            return final_str.strip()
        return final_str

    def render_message(
        self,
        message: Message,
        last: Optional[bool] = None,
        with_system: bool = True,
        with_contexts: bool = True,
    ) -> str:
        """
        Render a single message round (system, contexts, input, response)
        """
        input_str = self.render_instruction(
            message.input, message.contexts, with_system=with_system, with_contexts=with_contexts
        )
        response_str = self.render_response(message.response, last=last)

        final_str = self.format.join_messages([input_str, response_str])
        if self.format.BOS:
            final_str = self.format.BOS + final_str
        return final_str

    def render_instruction(
        self,
        text: str,
        contexts: Optional[List[str]] = None,
        with_system: bool = True,
        with_contexts: bool = True,
    ) -> str:
        """
        Render instruction components (system, contexts, input)
        """
        instruction: List[str] = []
        inputs: List[str] = []
        system_message = ""
        if with_system and self.system_message:
            system_message = self.render_system()
            if self.format.system_nested:
                # System message nested within input formatting (e.g. Llama2)
                inputs.append(system_message)
            else:
                instruction.append(system_message)

        if with_contexts and contexts:
            contexts_str = self.render_contexts(contexts)
            if self.format.contexts_nested:
                inputs.append(contexts_str)
            else:
                instruction.append(contexts_str)

        input_str = self.input_str(text)
        inputs.append(input_str)

        input_str = self.format.join_contents(inputs)
        input_str = self.format.input.render(input_str)
        instruction.append(input_str)
        return self.format.join_messages(instruction)

    def render_response(
        self,
        text: Optional[str],
        last: Optional[bool] = None,
        with_prefix: bool = True,
        with_suffix: bool = True,
    ) -> str:
        if text is None:
            # Empty response.
            if last is None:
                # Assume null response indicates last message unless specified
                last = True
            response_partial = self.response_str("")
            response_partial = self.format.response.render(
                response_partial, with_prefix=with_prefix, with_suffix=(with_suffix and not last)
            )
            if not last and self.format.EOS:
                response_partial += self.format.EOS
            return response_partial

        # Fully formatted response
        response_str = self.response_str(text)
        response_str = self.format.response.render(
            response_str, with_prefix=with_prefix, with_suffix=with_suffix
        )
        if self.format.EOS:
            response_str += self.format.EOS
        return response_str

    def render_system(self) -> str:
        return self.format.system.render(self.system_message)

    def render_examples(self, with_system: bool = True, with_contexts: bool = True) -> str:
        return self.render(
            self.examples, with_system=with_system, with_contexts=with_contexts, with_examples=False
        )

    def render_contexts(self, contexts: List[str]) -> str:
        contexts_str = self.format.join_contents([
            self.context_str(context, index=i)
            for i, context in enumerate(contexts)
        ])
        return self.format.contexts.render(contexts_str)

    def input_str(self, text: str, **kwargs) -> str:
        return self.input_template.format_map({"text": text, **kwargs})

    def context_str(self, text: str, **kwargs) -> str:
        return self.context_template.format_map({"text": text, **kwargs})

    def response_str(self, text: str, **kwargs) -> str:
        return self.response_template.format_map({"text": text, **kwargs})


@dataclass
class Conversation:
    messages: List[Message] = field(default_factory=list)

    message_retention: int = 5
    context_retention: int = 1

    def add(self, *messages: Message):
        """
        Add a message to the conversation
        """
        existing_count = len(self.messages)
        self.messages.extend(messages)
        # Trim old messages
        if self.message_retention > 0 and len(self.messages) > self.message_retention:
            removed_count = len(self.messages) - self.message_retention
            existing_count = existing_count - removed_count if existing_count >= removed_count else 0
            self.messages = self.messages[-self.message_retention:]

        # Remove contexts from old messages
        if self.context_retention > 0 and len(self.messages) > self.context_retention:
            for message in self.messages[existing_count-1:-self.context_retention]:
                message.contexts = []

    def clear(self):
        self.messages = []

    def render(self, prompt: Prompt, **kwargs) -> str:
        """
        Render the conversation with a prompt
        """
        return prompt.render(self.messages, **kwargs)
