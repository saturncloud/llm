from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from llm.prompt import Prompt, Message


@dataclass
class ZeroShotQA(Prompt):
    system_message: str = "Given the following contexts and a question, create a final answer. Only use the given context to arrive at your answer. If you don't know the answer, just say that you don't know. Don't make up an answer."
    context_template: str = "Context: {text}"
    input_template: str = "Question: {text}"


@dataclass
class FewShotQA(ZeroShotQA):
    examples: List[Message] = field(default_factory=lambda: [
        Message(
            contexts=[
                "No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.11.7 Severability.",
                "This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an injunction or other relief to protect its Intellectual Property Rights.",
                "(b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,",
            ],
            input="Which state/country's law governs the interpretation of the contract?",
            response="This Agreement is governed by English law.",
        ),
        Message(
            contexts=[
                "Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans. Last year COVID-19 kept us apart. This year we are finally together again",
                "More support for patients and families. To get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. It's based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  ARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer's, diabetes, and more.",
            ],
            input="What did the president say about Michael Jackson?",
            response="I don't know"
        ),
        Message(
            contexts=[
                "Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans. Last year COVID-19 kept us apart. This year we are finally together again.",
                "More support for patients and families. To get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. It's based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  ARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer's, diabetes, and more.",
            ],
            input="What is the purpose of ARPA-H?",
            response="The Advanced Research Projects Agency for Health will drive breakthroughs in cancer, Alzheimer's, diabetes, and more."
        ),
    ])


@dataclass
class StandaloneQuestion(Prompt):
    system_message: str = "Given the following conversation and a followup question, rephrase the followup question as a standalone question with any relevant context. The final answer should be a concise search query that will find results relevant to the followup question."
    input_template: str = "Followup Question: {text}"
    response_template: str = "Standalone Question: {text}"
    context_prompt: Prompt = field(default_factory=lambda: Prompt(
        input_template="Question: {text}",
        response_template="Answer: {text}",
    ))
    examples: List[Message] = field(default_factory=lambda: [
        Message(
            contexts=[
                "What is the purpose of ARPA-H?",
                "The Advanced Research Projects Agency for Health will drive breakthroughs in cancer, Alzheimer's, diabetes, and more.",
            ],
            input="When was it created?",
            response="When was the Advanced Research Projects Agency for Health created?"
        ),
        Message(
            contexts=[
                "What is the purpose of ARPA-H?",
                "The Advanced Research Projects Agency for Health will drive breakthroughs in cancer, Alzheimer's, diabetes, and more.",
                "When was it created?",
                "The Advanced Research Projects Agency for Health was created on March 15, 2022",
            ],
            input="What are some of the other breakthroughs?",
            response="In addition to cancer, Alzheimer's, and diabetes, what are some other breakthroughs driven by the Advanced Research Projects Agency for Health?",
        ),
        Message(
            contexts=[
                "Which state/country's law governs the interpretation of the contract?",
                "This Agreement is governed by English law.",
            ],
            input="Who are the primary parties bound by the contract?",
            response="Who are the primary parties bound by the contract?",
        ),
    ])

    @property
    def stop_strings(self) -> List[str]:
        # Sometimes models will generate an "Answer: ..." after the standalone question.
        # Use the context response template as an additional stop string to prevent this.
        stop_strings = super().stop_strings
        answer_label = self.context_prompt.response_str("")
        if answer_label.strip():
            stop_strings.append(answer_label)
        return stop_strings

    def render_contexts(self, contexts: List[str]) -> str:
        # Override to render contexts as a conversation
        messages: List[Message] = []
        for question, answer in zip(contexts[::2], contexts[1::2]):
            messages.append(Message(question, answer))
        return super().render_contexts([self.context_prompt.render(messages, strip=False)])
