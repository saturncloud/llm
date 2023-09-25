from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Role:
    prefix: str = ""
    suffix: str = ""

    def render(self, text: str, partial: bool = False) -> str:
        text = f"{self.prefix}{text}"
        if partial:
            return text
        return f"{text}{self.suffix}"

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
    user: Role = field(default_factory=Role)
    assistant: Role = field(default_factory=Role)
    system: Role = field(default_factory=Role)
    context: Role = field(default_factory=Role)
    example: Role = field(default_factory=Role)

    # System message nested within the first user message (Llama2 format)
    system_nested: bool = False
    # Separator between each message and context
    message_separator: str = "\n"
    # String added before each message input
    BOS: str = ""
    # String added after each message response
    EOS: str = ""

    def join(self, texts: List[str]) -> str:
        return self.message_separator.join(texts)

    @property
    def stop_strings(self) -> List[str]:
        return (
            self.user.stop_strings
            + self.assistant.stop_strings
            + self.system.stop_strings
            + self.context.stop_strings
            + self.example.stop_strings
        )


@dataclass
class UserAssistantFormat(PromptFormat):
    user: Role = field(default_factory=lambda: Role("User: "))
    assistant: Role = field(default_factory=lambda: Role("Assistant: "))


@dataclass
class RedpajamaFormat(PromptFormat):
    user: Role = field(default_factory=lambda: Role("<human>: "))
    assistant: Role = field(default_factory=lambda: Role("<bot>: "))


@dataclass
class VicunaFormat(PromptFormat):
    user: Role = field(default_factory=lambda: Role("USER: "))
    assistant: Role = field(default_factory=lambda: Role("ASSISTANT: "))
    BOS: str = "<s>"
    EOS: str = "</s>"


@dataclass
class Llama2Format(PromptFormat):
    user: Role = field(default_factory=lambda: Role(prefix="[INST] ", suffix=" [/INST]"))
    system: Role = field(default_factory=lambda: Role(prefix="<<SYS>>\n", suffix="\n<</SYS>>\n\n"))
    system_nested: bool = True
    message_separator: str = ""
    BOS: str = "<s>"
    EOS: str = "</s>"


@dataclass
class TogetherLlama2Format(PromptFormat):
    # Simplified version of Llama2 roles used in togethercomputer/Llama-2-7B-32K-Instruct
    user: Role = field(default_factory=lambda: Role(prefix="  [INST]  ", suffix="  [/INST]  "))
    system_nested: bool = True
    message_separator: str = ""


@dataclass
class Message:
    input: str
    response: Optional[str] = None
    contexts: List[str] = field(default_factory=list)


@dataclass
class Prompt:
    system_message: str = ""
    examples: List[List[Message]] = field(default_factory=list)

    # Applies to message input string separate from contexts
    input_template: str = "{input}"
    # Applies to individual contexts
    context_template: str = "{context}"
    # Applies to previous responses, and appended with "" to prompt the next response
    response_template: str = "{response}"

    def render(
        self,
        messages: List[Message],
        format: Optional[PromptFormat] = None,
        with_system: bool = True,
        with_contexts: bool = True,
        with_examples: bool = True,
    ) -> str:
        """
        Render the prompt as a conversation.
        """
        if format is None:
            format = PromptFormat()
        results: List[str] = []

        # Format examples
        if with_examples and self.examples:
            for example in self.examples:
                example_str = self.render(
                    example,
                    format=format,
                    with_system=with_system,
                    with_contexts=with_contexts,
                    with_examples=False,
                )
                example_str = format.example.render(example_str)
                results.append(example_str)
                # Only add system message once
                with_system = False

        num_messages = len(messages)
        if num_messages == 0 and len(results) == 0:
            # No messages, and no examples. Return system message.
            if with_system:
                system_message = format.system.render(self.system_message)
                if format.system_nested:
                    system_message = format.user.render(system_message)
                return system_message
            return ""

        # Format conversation
        for i, message in enumerate(messages):
            last = (i == num_messages - 1)
            message_str = self.render_message(
                message,
                format=format,
                last=last,
                with_system=(with_system and (i == 0)),
                with_contexts=with_contexts,
            )
            results.append(message_str)

        return format.join(results).strip()

    def render_message(
        self,
        message: Message,
        format: Optional[PromptFormat] = None,
        last: bool = False,
        with_system: bool = True,
        with_contexts: bool = True,
    ) -> str:
        if format is None:
            format = PromptFormat()
        results = []
        inputs = []

        # Format system message
        if with_system:
            system_message = format.system.render(self.system_message)
            if format.system_nested:
                # System message nested within input formatting (e.g. Llama2)
                inputs.append(system_message)
            else:
                # System message before input formatting
                results.append(system_message)

        # Format contexts
        if with_contexts:
            context_str = format.join([
                self.context_template.format(context=context)
                for context in message.contexts
            ])
            context_str = format.context.render(context_str)
            inputs.append(context_str)

        # Format user input
        user_input_str = self.input_template.format(input=message.input)
        inputs.append(user_input_str)

        # Combine input parts
        full_input_str = format.user.render(format.join(inputs))
        results.append(full_input_str)

        # Format response text
        if message.response is not None:
            # Add fully formatted response
            response_text = self.response_template.format(response=message.response)
            response_text = format.assistant.render(response_text)
            if format.EOS:
                response_text += format.EOS
            results.append(response_text)
        elif last:
            # Adds assistant prefix and response template to prompt a response
            # EOS is expected to be generated by the model once its response is complete
            response_partial = self.response_template.format(response="")
            response_partial = format.assistant.render(response_partial, partial=True)
            if response_partial:
                results.append(response_partial)

        # Combine all message parts
        final_str = format.join(results)
        if format.BOS:
            final_str = format.BOS + final_str
        return final_str


@dataclass
class Conversation:
    messages: List[Message] = field(default_factory=list)
    format: PromptFormat = field(default_factory=UserAssistantFormat)

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

    def render(self, prompt: Prompt) -> str:
        return prompt.render(self.messages, format=self.format)
