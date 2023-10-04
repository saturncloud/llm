from llm.training.dataprep import Concatenator


def test_concat():
    concat = Concatenator(chunk_size=5, total_number_of_examples=3)
    test_data = {
        "input_ids": [list(range(3)), list(range(3)), list(range(3))],
        "attention_mask": [list(range(3)), list(range(3)), list(range(3))],
        "lables": [list(range(3)), list(range(3)), list(range(3))],
    }
    result = concat(test_data, [0, 1, 2])
    breakpoint()
