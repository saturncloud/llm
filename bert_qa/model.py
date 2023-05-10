from token import RPAR
from typing import List, Optional, Tuple, Type, Union

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast, PreTrainedModel

# Other models to try:
# - deepset/roberta-large-squad2
# - deepset/xlm-roberta-large-squad2
# - deepset/tinyroberta-squad2
# - microsoft/layoutlmv2-large-uncased (or base)

DEFAULT_MODEL = "bert-large-uncased-whole-word-masking-finetuned-squad"


class BertQA:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerFast

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        model_cls: Union[Type[PreTrainedModel], Type[AutoModelForQuestionAnswering]] = AutoModelForQuestionAnswering,
        tokenizer_cls: Union[Type[PreTrainedTokenizerFast], Type[AutoTokenizer]] = AutoTokenizer,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer_cls.from_pretrained(model)
        self.model = model_cls.from_pretrained(model).to(self.device)

    def best_answer(self, question: str, contexts: List[str], **kwargs) -> Tuple[str, float, int]:
        answers = self.topk_answers(question, contexts, top_k=1, **kwargs)
        if len(answers) == 0:
            return "", -1e20, -1
        return answers[0]

    def topk_answers(self, question: str, contexts: List[str], top_k: int = 3, **kwargs) -> List[Tuple[str, float, int]]:
        scores = torch.full((1, len(contexts)), float(-1e20), dtype=torch.float64)
        answers = [""] * len(contexts)

        for i, context in enumerate(contexts):
            answers[i], scores[0, i] = self.answer_question(question, context, **kwargs)

        top_scores, top_indices = torch.topk(scores, min(top_k, len(contexts)), 1)
        return [
            (answers[i], score, i)
            for i, score in zip(top_indices[0].tolist(), top_scores[0].tolist())
        ]

    def answer_question(
        self,
        question: str,
        context: str,
        span_length: int = 512,
        span_overlap: int = 64,
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
            return_attention_mask=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        ).to(self.device)

        input_ids = inputs.input_ids.to(self.device)
        token_type_ids = inputs.token_type_ids.to(self.device) if hasattr(inputs, "token_type_ids") else None
        attention_mask = inputs.attention_mask.to(self.device)

        # Run model
        with torch.no_grad():
            output = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        answer_start_scores = output["start_logits"]
        answer_end_scores = output["end_logits"]

        # Set low scores for CLS token
        answer_start_scores[:, 0] = -1e20
        answer_end_scores[:, 0] = -1e20

        # Best answers in each span
        batch_start_indices = torch.argmax(answer_start_scores, 1)
        batch_end_indices = torch.argmax(answer_end_scores, 1)
        batch_start_scores = torch.tensor([
            scores[batch_start_indices[i]] for i, scores in enumerate(answer_start_scores)
        ], device=self.device)
        batch_end_scores = torch.tensor([
            scores[batch_end_indices[i]] for i, scores in enumerate(answer_end_scores)
        ], device=self.device)

        # Combine start and end scores to get an approximation for the whole answer
        # Weighted average favoring start score
        batch_avg_scores = (1.5 * batch_start_scores + 0.5 * batch_end_scores) / 2
        best_batch = torch.argmax(batch_avg_scores)

        # Retrieve token IDs for the answer
        start_index = batch_start_indices[best_batch]
        end_index = batch_end_indices[best_batch]

        answer = self.extract_answer(context, inputs, best_batch, start_index, end_index)
        return answer, float(batch_avg_scores[best_batch])

    def extract_answer(self, context: str, inputs: BatchEncoding, batch: int, start: int, end: int) -> str:
        answer_ids = inputs.input_ids[batch, start:end+1]
        # If answer starts with CLS token, then the question is included. Remove it.
        if len(answer_ids) > 0 and answer_ids[0] == self.tokenizer.cls_token_id:
            sep_idx = 0
            for i, id in enumerate(answer_ids):
                if id == self.tokenizer.sep_token_id:
                    sep_idx = i
                    break
            start += sep_idx + 1
            answer_ids = answer_ids[sep_idx+1:]

        # Extract answer from the original text rather than decoding IDs for better readability
        answer_start_offset = int(inputs.offset_mapping[batch, start][0])
        answer_end_offset = int(inputs.offset_mapping[batch, end][-1])
        return context[answer_start_offset:answer_end_offset]

    def tokenize(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchEncoding:
        return self.tokenizer(text, text_pair, add_special_tokens=add_special_tokens, return_tensors=return_tensors, **kwargs)
