from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import BatchEncoding, BertTokenizerFast, BertForQuestionAnswering

from bert_qa.docs import load_docs


@dataclass
class Answer:
    text: str
    question: str
    source: str
    score: float

    def __str__(self) -> str:
        return self.text

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def print(self):
        print(
            f"Question: {self.question}",
            f"\nAnswer: {self.text}",
            f"\nSource: {self.source}",
            f"\nScore: {self.score}\n",
        )


class BertQA:
    def __init__(self):
        self.docs = load_docs()
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.model: BertForQuestionAnswering = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    def search_docs(self, question: str, section: Optional[str] = None, span_length: int = 512, span_overlap: int = 128) -> Answer:
        if not section:
            best_section_score = -1e20
            best_section = ""
            sections = self.docs.keys()
            for section in sections:
                _, score = self.answer_question(question, section)
                if score > best_section_score:
                    best_section_score = score
                    best_section = section
            section = best_section

        best_score = -1e20
        best_answer = ""
        best_source = ""
        for file, context in self.docs[section].items():
            answer, score = self.answer_question(question, context, span_length=span_length, span_overlap=span_overlap)
            if answer and score > best_score:
                best_answer = answer
                best_source = file
                best_score = score
        return Answer(best_answer, question, best_source, best_score)

    def answer_question(
        self,
        question: str,
        context: str,
        span_length: int = 512,
        span_overlap: int = 128,
    ) -> Tuple[str, float]:
        assert span_length <= 512, "max sequence length for the BERT model is 512"
        assert span_length > span_overlap, "span_length must be greater than span_overlap"
        assert span_length > 0
        assert span_overlap > 0


        # Tokenize input
        inputs = self.tokenize(
            question,
            context,
            max_length=span_length,
            padding=True,
            truncation="only_second",
            stride=span_overlap,
            return_overflowing_tokens=True,
        )

        # Run model
        output = self.model(inputs.input_ids, token_type_ids=inputs.token_type_ids)
        answer_start_scores = output["start_logits"]
        answer_end_scores = output["end_logits"]

        # TODO: Remove answers that are in the qusetion (check via token_type_ids)
        #       before determining best answer.
        # Best answers in each span
        batch_start_indices = torch.argmax(answer_start_scores, 1)
        batch_end_indices = torch.argmax(answer_end_scores, 1)
        batch_start_scores = torch.tensor([
            scores[batch_start_indices[i]] for i, scores in enumerate(answer_start_scores)
        ])
        batch_end_scores = torch.tensor([
            scores[batch_end_indices[i]] for i, scores in enumerate(answer_end_scores)
        ])

        # Combine start and end scores to get an approximation for the whole answer
        # Weighted average favoring start score
        batch_avg_scores = (1.5 * batch_start_scores + 0.5 * batch_end_scores) / 2
        best_batch = torch.argmax(batch_avg_scores)

        # Retrieve token IDs for the answer
        start_index = batch_start_indices[best_batch]
        end_index = batch_end_indices[best_batch]
        answer_ids = inputs.input_ids[best_batch, start_index:end_index+1]

        # If answer starts with CLS token, then the question is included. Remove it.
        if len(answer_ids) > 0 and answer_ids[0] == self.tokenizer.cls_token_id:
            sep_idx = 0
            for i, id in enumerate(answer_ids):
                if id == self.tokenizer.sep_token_id:
                    sep_idx = i
                    break
            answer_ids = answer_ids[sep_idx+1:]

        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return answer, float(batch_avg_scores[best_batch])

    def tokenize(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchEncoding:
        return self.tokenizer(text, text_pair, add_special_tokens=add_special_tokens, return_tensors=return_tensors, **kwargs)
