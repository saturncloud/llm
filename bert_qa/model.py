from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from transformers import BatchEncoding, BertTokenizerFast, BertForQuestionAnswering

from bert_qa.docs import load_docs


@dataclass
class Answer:
    question: str
    text: str
    source: str
    score: float

    def __str__(self) -> str:
        return self.text

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BertQA:
    def __init__(self):
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.model: BertForQuestionAnswering = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        self.docs = self.load_docs_tokenized(skip={"enterprise", "examples", "release-notes"})

    def search_docs(self, question: str, span_length: int = 512, span_overlap: int = 128) -> Answer:
        # Only searching the concepts section for now
        section = "concepts"
        for file, context in self.docs[section].items():
            answer, score = self.answer_question(question, context, span_length=span_length, span_overlap=span_overlap)
            if answer and score > best_score:
                best_answer = answer
                best_source = f"{section}/{file}"
                best_score = score
        return Answer(question, best_answer, best_source, best_score)

    def answer_question(
        self,
        question: str,
        context: Union[str, BatchEncoding],
        span_length: int = 512,
        span_overlap: int = 128,
    ) -> Tuple[str, float]:
        assert span_length <= 512, "max sequence length for the BERT model is 512"
        assert span_length > span_overlap, "span_length must be greater than span_overlap"
        assert span_length > 0
        assert span_overlap > 0

        # Tokenize input
        question_inputs = self.tokenize(question).input_ids
        if isinstance(context, str):
            context_inputs = self.tokenize(context).input_ids
        else:
            context_inputs = context.input_ids

        # Strip CLS token from start of context
        if context_inputs[0][0] == self.tokenizer.cls_token_id:
            context_inputs = context_inputs[:, 1:]

        question_length = len(question_inputs[0])
        context_length = len(context_inputs[0])
        context_window = span_length - question_length
        step_size = min(span_length - span_overlap, context_window)

        assert question_length < span_length, "span_length must be greater than the length of the question"

        # Break input into batches with a sliding window over the context
        # Each batch includes the full question with a piece of the context
        input_batches = torch.tensor([], dtype=torch.int64)
        token_type_id_batches = torch.tensor([], dtype=torch.int64)
        i = 0
        while i < context_length:
            window_end = min(i + context_window, context_length)
            num_tokens = window_end - i
            padding = torch.full([1, context_window - num_tokens], self.tokenizer.pad_token_id)
            window_inputs = torch.cat((question_inputs, context_inputs[:, i:window_end], padding), 1)
            token_type_ids = torch.cat((torch.full([1, question_length], 0), torch.full([1, span_length - question_length], 1)), 1)

            input_batches = torch.cat((input_batches, window_inputs), 0)
            token_type_id_batches = torch.cat((token_type_id_batches, token_type_ids))
            i += step_size

        # Run model
        output = self.model(input_batches, token_type_ids=token_type_id_batches)
        answer_start_scores = output["start_logits"]
        answer_end_scores = output["end_logits"]

        # Determine best matching answer
        batch_start_indices = torch.argmax(answer_start_scores, 1)
        batch_end_indices = torch.argmax(answer_end_scores, 1)
        batch_start_scores = torch.tensor([
            scores[batch_start_indices[i]] for i, scores in enumerate(answer_start_scores)
        ])
        batch_end_scores = torch.tensor([
            scores[batch_end_indices[i]] for i, scores in enumerate(answer_end_scores)
        ])

        batch_avg_scores = (batch_start_scores + batch_end_scores) / 2
        best_batch = torch.argmax(batch_avg_scores)

        start_index = batch_start_indices[best_batch]
        end_index = batch_end_indices[best_batch]
        answer_ids = input_batches[best_batch, start_index:end_index+1]

        # If answer starts with CLS token, then the question is included. Remove it.
        if len(answer_ids) > 0 and answer_ids[0] == self.tokenizer.cls_token_id:
            answer_ids = answer_ids[question_length:]
        # Remove any other special tokens from the answer
        for special_id in self.tokenizer.all_special_ids:
            answer_ids = answer_ids[answer_ids != special_id]

        answer = self.tokenizer.decode(answer_ids)
        if answer.startswith(question):
            answer = answer[len(question):]

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

    def load_docs_tokenized(self, skip: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, torch.Tensor]]:
        docs = load_docs(skip=skip)
        tokenized: Dict[str, Dict[str, torch.Tensor]] = {}
        for section, files in docs.items():
            tokenized[section] = {}
            for file, content in files.items():
                tokenized[section][file] = self.tokenize(content)
        return tokenized