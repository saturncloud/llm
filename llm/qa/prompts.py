from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from llm.prompt import Prompt, Message, PromptFormat


@dataclass
class ZeroShotQA(Prompt):
    system_message: str = "Given the following contexts and a question, create a final answer. Only use the given context to arrive at your answer. If you don't know the answer, just say that you don't know. Don't make up an answer."
    context_template: str = "\nContext: {context}"
    input_template: str = "\nQuestion: {input}"


@dataclass
class FewShotQA(ZeroShotQA):
    examples: List[List[Message]] = field(default_factory=lambda: [
        [
            Message(
                contexts=[
                    "No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.11.9 No Third-Party Beneficiaries.",
                    "This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an injunction or other relief to protect its Intellectual Property Rights.",
                    "(b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,",
                ],
                input="Which state/country's law governs the interpretation of the contract?",
                response="This Agreement is governed by English law.",
            ),
        ],
        [
            Message(
                contexts=[
                    "Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans. Last year COVID-19 kept us apart. This year we are finally together again. Tonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. With a duty to one another to the American people to the Constitution. And with an unwavering resolve that freedom will always triumph over tyranny. Six days ago, Russia's Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. He thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. He met the Ukrainian people. From President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. Groups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.",
                    "More support for patients and families. To get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. It's based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  ARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer's, diabetes, and more. A unity agenda for the nation. We can do this. My fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. In this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. We have fought for freedom, expanded liberty, defeated totalitarianism and terror. And built the strongest, freest, and most prosperous nation the world has ever known. Now is the hour. Our moment of responsibility. Our test of resolve and conscience, of history itself. It is in this moment that our character is formed. Our purpose is found. Our future is forged. Well I know this nation.",
                ],
                input="What did the president say about Michael Jackson?",
                response="I don't know"
            ),
            Message(
                contexts=[
                    "Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans. Last year COVID-19 kept us apart. This year we are finally together again. Tonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. With a duty to one another to the American people to the Constitution. And with an unwavering resolve that freedom will always triumph over tyranny. Six days ago, Russia's Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. He thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. He met the Ukrainian people. From President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. Groups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.",
                    "More support for patients and families. To get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. It's based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  ARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer's, diabetes, and more. A unity agenda for the nation. We can do this. My fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. In this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. We have fought for freedom, expanded liberty, defeated totalitarianism and terror. And built the strongest, freest, and most prosperous nation the world has ever known. Now is the hour. Our moment of responsibility. Our test of resolve and conscience, of history itself. It is in this moment that our character is formed. Our purpose is found. Our future is forged. Well I know this nation.",
                ],
                input="What is the purpose of ARPA-H?",
                response="The Advanced Research Projects Agency for Health will drive breakthroughs in cancer, Alzheimer's, diabetes, and more."
            ),
        ],
    ])


@dataclass
class StandaloneQuestion(Prompt):
    system_message: str = "Given the following conversation and a follow up question, rephrase the follow up question with any relevant context. The final answer should be a concise search query that will find results relevant to the original question."
    input_template: str = "\nQuestion: {input}"
    response_template: str = "\nAnswer: {response}"
    final_input_template: str = "\nFollowup Question: {input}"
    final_response_template: str = "\nRephrased Question: {response}"
    examples: List[List[Message]] = field(default_factory=lambda: [
        [
            Message(
                input="What is the purpose of ARPA-H?",
                response="The Advanced Research Projects Agency for Health will drive breakthroughs in cancer, Alzheimer's, diabetes, and more.",
            ),
            Message(
                input="When was it created?",
                response="When was the Advanced Research Projects Agency for Health created?"
            ),
        ],
        [
            Message(
                input="What is the purpose of ARPA-H?",
                response="The Advanced Research Projects Agency for Health will drive breakthroughs in cancer, Alzheimer's, diabetes, and more.",
            ),
            Message(
                input="When was it created?",
                response="The Advanced Research Projects Agency for Health was created on March 15, 2022",
            ),
            Message(
                input="What are some of the other breakthroughs?",
                response="In addition to cancer, Alzheimer's, and diabetes, what are some other breakthroughs driven by the Advanced Research Projects Agency for Health?",
            ),
        ],
        [
            Message(

                input="Which state/country's law governs the interpretation of the contract?",
                response="This Agreement is governed by English law.",
            ),
            Message(
                input="Who are the primary parties bound by the contract?",
                response="Who are the primary parties bound by the contract?",
            ),
        ],
    ])

    def render_message(
        self,
        format: PromptFormat,
        message: Message,
        index: int = 0,
        last: bool = False,
        **kwargs
    ) -> str:
        print(index, last)
        input_template = self.input_template
        response_template = self.response_template
        if last:
            self.input_template = self.final_input_template
            self.response_template = self.final_response_template
        result = super().render_message(format, message, index, last, **kwargs)
        if last:
            self.input_template = input_template
            self.response_template = response_template
        return result
