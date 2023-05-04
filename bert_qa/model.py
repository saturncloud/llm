from typing import List, Optional, Tuple

import torch
from transformers import BatchEncoding, BertTokenizerFast, BertForQuestionAnswering, RobertaForQuestionAnswering, RobertaTokenizerFast


class BertQA:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.tokenizer: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained("deepset/tinyroberta-squad2")
        # self.model: RobertaForQuestionAnswering = RobertaForQuestionAnswering.from_pretrained("deepset/tinyroberta-squad2").to(self.device)

        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        self.model: BertForQuestionAnswering = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad").to(self.device)

    def best_answer(self, question: str, contexts: List[str], **kwargs) -> Tuple[str, float, int]:
        best_score = -1e20
        best_answer = ""
        best_context = -1

        for i, context in enumerate(contexts):
            answer, score = self.answer_question(question, context, **kwargs)
            if answer and score > best_score:
                best_answer = answer
                best_context = i
                best_score = score
        return best_answer, best_score, best_context

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
        output = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        answer_start_scores = output["start_logits"]
        answer_end_scores = output["end_logits"]

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

        # Retrieve answer_ids from inputs in CPU memory
        answer_ids = inputs.input_ids[best_batch, start_index:end_index+1]

        # If answer starts with CLS token, then the question is included. Remove it.
        if len(answer_ids) > 0 and answer_ids[0] == self.tokenizer.cls_token_id:
            sep_idx = 0
            for i, id in enumerate(answer_ids):
                if id == self.tokenizer.sep_token_id:
                    sep_idx = i
                    break
            start_index += sep_idx + 1
            answer_ids = answer_ids[sep_idx+1:]

        # Extract answer from the original text rather than decoding IDs for better readability
        answer_start_offset = int(inputs.offset_mapping[best_batch, start_index][0])
        answer_end_offset = int(inputs.offset_mapping[best_batch, end_index][-1])
        answer = context[answer_start_offset:answer_end_offset]

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
