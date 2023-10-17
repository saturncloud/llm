from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Optional

if TYPE_CHECKING:
    from llm.model_configs import ModelConfig


@dataclass
class Role:
    prefix: str = ""
    suffix: str = ""
    strip_text: bool = True

    def render(self, text: str, with_prefix: bool = True, with_suffix: bool = True) -> str:
        if self.strip_text:
            text = text.strip()
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
    nested_system: bool = False
    nested_contexts: bool = True

    # Strip prefixes/suffixes before returning them as strop strings
    strip_stop_strings: bool = True
    # Separator between roles
    role_separator: str = "\n"
    # Separator between contents of a role (e.g. contexts and nested components of the input role)
    content_separator: str = "\n"
    # String added before each message input
    BOS: str = ""
    # String added after each message response
    EOS: str = ""

    def join_roles(self, texts: Iterable[str]) -> str:
        return self.role_separator.join(texts)

    def join_contents(self, texts: Iterable[str]) -> str:
        return self.content_separator.join(texts)

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
    """
    <s>[INST] <<SYS>>
    {system_message}
    <</SYS>>

    {input} [/INST] {response} </s>
    """

    input: Role = field(default_factory=lambda: Role(prefix="[INST] ", suffix=" [/INST]"))
    system: Role = field(default_factory=lambda: Role(prefix="<<SYS>>\n", suffix="\n<</SYS>>\n\n"))
    contexts: Role = field(default_factory=lambda: Role(suffix="\n"))
    response: Role = field(default_factory=lambda: Role(prefix=" ", suffix=" "))
    nested_system: bool = True
    role_separator: str = ""
    BOS: str = "<s>"
    EOS: str = "</s>"


@dataclass
class TogetherLlama2Format(PromptFormat):
    """
    Simplified version of Llama2 roles used in togethercomputer/Llama-2-7B-32K-Instruct
    The exact formatting differs a bit from training data and model card, this is a mix of the two
    that does well for inference.
    """

    input: Role = field(default_factory=lambda: Role(prefix="[INST]  ", suffix="  [/INST]\n\n"))
    nested_system: bool = True
    role_separator: str = ""


@dataclass
class ChatMLFormat(PromptFormat):
    input: Role = field(
        default_factory=lambda: Role(prefix="<|im_start|>user\n", suffix="<|im_end|>")
    )
    response: Role = field(
        default_factory=lambda: Role(prefix="<|im_start|>assistant\n", suffix="<|im_end|>")
    )
    system: Role = field(
        default_factory=lambda: Role(prefix="<|im_start|>system\n", suffix="<|im_end|>")
    )


@dataclass
class DollyFormat(PromptFormat):
    input: Role = field(default_factory=lambda: Role(prefix="\n### Instruction:\n"))
    response: Role = field(default_factory=lambda: Role(prefix="\n### Response:\n"))


@dataclass
class Message:
    input: str
    response: Optional[str] = None
    contexts: Optional[List[str]] = None


@dataclass
class Prompt:
    system_message: str = ""
    examples: List[Message] = field(default_factory=list)
    format: PromptFormat = field(default_factory=PromptFormat)

    # Templates are applied before role formatting
    text_key: str = "text"
    input_template: str = "{text}"
    context_template: str = "{text}"  # Also supports {index}
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
        Render the prompt over multiple message rounds
        Used for conversations or packed training examples
        """
        results: List[str] = []

        # Format examples
        if with_examples and self.examples:
            examples_str = self.render_examples(
                with_system=with_system, with_contexts=with_contexts
            )
            results.append(examples_str)
            # System messed is included with examples
            with_system = False

        num_messages = len(messages)
        if num_messages == 0 and len(results) == 0:
            # No messages, and no examples. Return system message.
            if with_system and self.system_message:
                system_message = self.render_system()
                if self.format.nested_system:
                    system_message = self.format.input.render(system_message)
                return system_message
            return ""

        # Format conversation
        for i, message in enumerate(messages):
            message_str = self.render_message(
                message,
                with_system=(with_system and (i == 0)),
                with_contexts=with_contexts,
            )
            results.append(message_str)

        final_str = self.format.join_roles(results)
        if strip:
            return final_str.strip()
        return final_str

    def render_message(
        self,
        message: Message,
        with_system: bool = True,
        with_contexts: bool = True,
    ) -> str:
        """
        Render a single message round (system, contexts, input, response)
        """
        instruction_str = self.render_instruction(
            message.input, message.contexts, with_system=with_system, with_contexts=with_contexts
        )
        response_str = self.render_response(message.response)
        return self.format.join_roles([instruction_str, response_str])

    def render_instruction(
        self,
        text: str,
        contexts: Optional[List[str]] = None,
        with_system: bool = True,
        with_contexts: bool = True,
        with_response_prefix: bool = False,
    ) -> str:
        """
        Render instruction components (system, contexts, input)
        """
        instruction: List[str] = []
        nested: List[str] = []
        system_message = ""
        if with_system and self.system_message:
            system_message = self.render_system()
            if self.format.nested_system:
                # System message nested within input formatting (e.g. Llama2)
                nested.append(system_message)
            else:
                instruction.append(system_message)

        if with_contexts and contexts:
            contexts_str = self.render_contexts(contexts)
            if self.format.nested_contexts:
                nested.append(contexts_str)
            else:
                instruction.append(contexts_str)

        input_str = self.input_str(text)
        nested.append(input_str)
        input_str = self.format.join_roles(nested)
        input_str = self.format.input.render(input_str)

        # Join remaining instruction roles and prepend BOS
        instruction.append(input_str)
        instruction_str = self.format.join_roles(instruction)
        if self.format.BOS:
            instruction_str = self.format.BOS + instruction_str
        if with_response_prefix:
            response_prefix = self.render_response(None, with_suffix=False)
            instruction_str = self.format.join_roles([instruction_str, response_prefix])
        return instruction_str

    def render_response(
        self,
        text: Optional[str],
        with_prefix: bool = True,
        with_suffix: bool = True,
    ) -> str:
        if text is None:
            # Empty response. Render empty string without suffix to prompt the model for completion
            response_partial = self.response_str("", with_prefix=with_prefix, with_suffix=False)
            return self.format.response.render(
                response_partial, with_prefix=with_prefix, with_suffix=False
            )

        # Fully formatted response
        response_str = self.response_str(text, with_prefix=with_prefix, with_suffix=with_suffix)
        response_str = self.format.response.render(
            response_str, with_prefix=with_prefix, with_suffix=with_suffix
        )
        if self.format.EOS:
            response_str += self.format.EOS
        return response_str

    def render_system(self) -> str:
        return self.format.system.render(self.system_message)

    def render_examples(
        self,
        examples: Optional[List[Message]] = None,
        with_system: bool = True,
        with_contexts: bool = True,
    ) -> str:
        examples_str = self.render(
            examples or self.examples,
            with_system=with_system,
            with_contexts=with_contexts,
            with_examples=False,
        )
        return self.format.examples.render(examples_str)

    def render_contexts(self, contexts: List[str]) -> str:
        contexts_str = self.format.join_contents(
            [self.context_str(context, index=i) for i, context in enumerate(contexts)]
        )
        return self.format.contexts.render(contexts_str)

    def input_str(self, text: str, **kwargs) -> str:
        return self.input_template.format_map({self.text_key: text, **kwargs})

    def context_str(self, text: str, **kwargs) -> str:
        return self.context_template.format_map({self.text_key: text, **kwargs})

    def response_str(
        self, text: str, with_prefix: bool = True, with_suffix: bool = True, **kwargs
    ) -> str:
        # Response template has to be handled specially since it may be incomplete
        template = self.response_template
        if not with_prefix or not with_suffix:
            split_key = "{" + self.text_key + "}"
            split = template.split(split_key)
            if not with_prefix:
                split[0] = ""
            if not with_suffix:
                split[-1] = ""
            template = split_key.join(split)

        return template.format_map({self.text_key: text, **kwargs})


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
            existing_count = (
                existing_count - removed_count if existing_count >= removed_count else 0
            )
            self.messages = self.messages[-self.message_retention :]

        # Remove contexts from old messages
        if self.context_retention > 0 and len(self.messages) > self.context_retention:
            for message in self.messages[existing_count - 1 : -self.context_retention]:
                message.contexts = None

    def clear(self):
        self.messages = []

    def render(self, prompt: Prompt, **kwargs) -> str:
        """
        Render the conversation with a prompt
        """
        return prompt.render(self.messages, **kwargs)


prompts = {}


@dataclass
class DefaultPrompt(Prompt):
    """
    A basic prompt that is usable (but not very good) at most tasks
    """
    input_template: str = "Input: {text}"
    context_template: str = "Context: {text}"
    response_template: str = "Response: {text}"
