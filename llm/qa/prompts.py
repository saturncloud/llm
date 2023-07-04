from __future__ import annotations
from copy import deepcopy

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from textwrap import dedent


@dataclass
class Prompt:
    """
    Formatting for basic LLM prompting
    """
    template: str = ""
    inputs: List[str] = field(default_factory=list)
    default_kwargs: Dict[str, Any] = field(default_factory=dict)
    dedent: bool = True

    def __post_init__(self):
        if self.dedent:
            self.template = dedent(self.template).strip()

    def render(self, **kwargs) -> str:
        kwargs = {
            **self.default_kwargs,
            **kwargs,
        }
        return self.template.format(**kwargs)


@dataclass
class ContextPrompt(Prompt):
    """
    Formatting for prompts that include additional contexts, such as the results
    of a semantic search.
    """
    default_context_label: str = "Context"

    def render(self, contexts: Optional[List[str]] = None, context_label: Optional[str] = None, **kwargs) -> str:
        context = "\n".join([f"{context_label}: {c.strip()}" for c in contexts or []])
        context_label = context_label or self.default_context_label

        if "context_label" in self.inputs:
            kwargs["context_label"] = context_label
        if "context" in self.inputs:
            # Context included in template
            kwargs["context"] = context
            append_context = False
        else:
            # Add context to end of template
            append_context = True

        prompt = super().render(**kwargs)
        if append_context:
            prompt += f"\n{context}"
        return prompt

ZERO_SHOT = ContextPrompt(
    template="""
    Given the following contexts and a question, create a final answer. Only use the given context to arrive at your answer. If you don't know the answer, just say that you don't know. Don't try to make up an answer.

    {context}
    {roles[0]}: {question}
    {roles[1]}:
    """,
    inputs=["context", "roles", "question"],
    default_kwargs={
        "roles": ["Question", "Answer"],
    },
)

FEW_SHOT = ContextPrompt(
    template="""
    Given the following contexts and a question, create a final answer. Only use the given context to arrive at your answer. If you don't know the answer, just say that you don't know. Don't try to make up an answer.

    {context_label}: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.11.9 No Third-Party Beneficiaries.
    {context_label}: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
    {context_label}: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
    {roles[0]}: Which state/country's law governs the interpretation of the contract?
    {roles[1]}: This Agreement is governed by English law.
    {roles[0]}:

    {context_label}: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans. Last year COVID-19 kept us apart. This year we are finally together again. Tonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. With a duty to one another to the American people to the Constitution. And with an unwavering resolve that freedom will always triumph over tyranny. Six days ago, Russia's Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. He thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. He met the Ukrainian people. From President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. Groups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.
    {context_label}: More support for patients and families. To get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. It's based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  ARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer's, diabetes, and more. A unity agenda for the nation. We can do this. My fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. In this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. We have fought for freedom, expanded liberty, defeated totalitarianism and terror. And built the strongest, freest, and most prosperous nation the world has ever known. Now is the hour. Our moment of responsibility. Our test of resolve and conscience, of history itself. It is in this moment that our character is formed. Our purpose is found. Our future is forged. Well I know this nation.
    {roles[0]}: What did the president say about Michael Jackson?
    {roles[1]}: I don't know.
    {roles[0]}: What is the purpose of ARPA-H?
    {roles[1]}: The Advanced Research Projects Agency for Health will drive breakthroughs in cancer, Alzheimer's, diabetes, and more.
    {roles[0]}:

    {context}
    {roles[0]}: {question}
    {roles[1]}:
    """,
    inputs=["roles", "context_label", "context", "question"],
    default_kwargs={
        "roles": ["Question", "Answer"],
    },
)

STANDALONE_QUESTION = Prompt(
    template="""
    Given the following conversation and a follow up question, rephrase the follow up question with any relevant context.

    {roles[0]}: What is the purpose of ARPA-H?
    {roles[1]}: The Advanced Research Projects Agency for Health will drive breakthroughs in cancer, Alzheimer's, diabetes, and more.
    {roles[0]}: When was it created?
    Standalone {roles[0]}: When was the Advanced Research Projects Agency for Health created?

    {roles[0]}: What is the purpose of ARPA-H?
    {roles[1]}: The Advanced Research Projects Agency for Health will drive breakthroughs in cancer, Alzheimer's, diabetes, and more.
    {roles[0]}: When was it created?
    {roles[1]}: The Advanced Research Projects Agency for Health was created on March 15, 2022
    {roles[0]}: What are some of the other breakthroughs?
    Standalone {roles[0]}: In addition to cancer, Alzheimer's, and diabetes, what are some other breakthroughs driven by the Advanced Research Projects Agency for Health?

    {roles[0]}: Which state/country's law governs the interpretation of the contract?
    {roles[1]}: This Agreement is governed by English law.
    {roles[0]}: Who are the primary parties bound by the contract?
    Standalone {roles[0]}: Who are the primary parties bound by the contract?

    {conversation}
    {roles[0]}: {question}
    Standalone {roles[0]}:
    """,
    inputs=["roles", "conversation", "question"],
    default_kwargs={
        "roles": ["Question", "Answer"],
    },
)

# Dolly style wrapper for other prompts
INSTRUCTION = Prompt(
    template="""
    Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction:
    {instruction}
    """,
    inputs=["instruction"],
)

INSTRUCTION_ZERO_SHOT = ContextPrompt(
    template=INSTRUCTION.render(instruction=ZERO_SHOT.template),
    inputs=deepcopy(ZERO_SHOT.inputs)
)

INSTRUCTION_FEW_SHOT = ContextPrompt(
    template=INSTRUCTION.render(instruction=FEW_SHOT.template),
    inputs=deepcopy(FEW_SHOT.inputs),
    default_kwargs=deepcopy(FEW_SHOT.default_kwargs),
)
