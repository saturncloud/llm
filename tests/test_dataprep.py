from pytest import mark

from llm.training.dataprep import Concatenator


def make_dummy_data(length: int, rows: int, start: int = 0):
    return {
        "input_ids": [list(range(start, start + length))] * rows,
        "attention_mask": [list(range(start, start + length))] * rows,
        "labels": [list(range(start, start + length))] * rows,
    }


def test_rows_to_process():
    concat = Concatenator(chunk_size=5, total_number_of_examples=4)
    concat.prepare_data_to_process(make_dummy_data(3, 4))
    assert concat.rows_to_process() == 4


def test_check_input_too_long():
    concat = Concatenator(chunk_size=5, total_number_of_examples=4)
    concat.to_process = make_dummy_data(10, 1)
    assert concat.check_for_input_too_long()

    concat = Concatenator(chunk_size=5, total_number_of_examples=4)
    concat.to_process = make_dummy_data(2, 1)
    assert not concat.check_for_input_too_long()


def test_is_last_output_full():
    concat = Concatenator(chunk_size=5, total_number_of_examples=4)
    concat.output = make_dummy_data(5, 1)
    assert concat.is_last_output_full()

    concat = Concatenator(chunk_size=5, total_number_of_examples=4)
    concat.output = make_dummy_data(4, 1)
    assert not concat.is_last_output_full()


def test_can_accumulate():
    concat = Concatenator(chunk_size=5, total_number_of_examples=4)
    concat.add_new_output_entry()
    concat.to_process = make_dummy_data(3, 1)
    assert concat.can_accumulate_without_exceeding_chunk_size()

    concat.output = make_dummy_data(1, 1)
    assert concat.can_accumulate_without_exceeding_chunk_size()

    concat.output = make_dummy_data(3, 1)
    assert not concat.can_accumulate_without_exceeding_chunk_size()


def test_pad_last_ouptut_row():
    concat = Concatenator(chunk_size=5, total_number_of_examples=4)
    concat.output = make_dummy_data(3, 1)
    concat.pad_last_output_row()
    assert concat.output["input_ids"][-1] == list(range(3)) + [0, 0]
    assert concat.output["attention_mask"][-1] == list(range(3)) + [0, 0]
    assert concat.output["labels"][-1] == list(range(3)) + [-100, -100]


def test_move_last_row_to_residual():
    concat = Concatenator(chunk_size=5, total_number_of_examples=4)
    data = Concatenator.concat(make_dummy_data(3, 2), make_dummy_data(4, 2))
    concat.output = data
    concat.move_last_output_row_to_residual()
    assert concat.output == Concatenator.concat(make_dummy_data(3, 2), make_dummy_data(4, 1))
    assert concat.residual == make_dummy_data(4, 1)


def test_concat():
    concat = Concatenator(chunk_size=5, total_number_of_examples=4)
    test_data = make_dummy_data(3, 4)
    result = concat(test_data, [0, 1, 2, 3])
    assert result == {
        "input_ids": [list(range(3)) + [0, 0]] * 4,
        "attention_mask": [list(range(3)) + [0, 0]] * 4,
        "labels": [list(range(3)) + [-100, -100]] * 4,
    }
    assert concat.residual == {"input_ids": [], "attention_mask": [], "labels": []}


def test_concat_multi_step():
    concat = Concatenator(chunk_size=5, total_number_of_examples=5)
    test_data = make_dummy_data(3, 2)
    result1 = concat(test_data, [0, 1])
    test_data = make_dummy_data(3, 3)
    result2 = concat(test_data, [2, 3, 4])

    assert len(result1["input_ids"] == 1)
    assert len(result2["input_ids"] == 4)


def test_concat_residual():
    concat = Concatenator(chunk_size=5, total_number_of_examples=5)
    test_data = make_dummy_data(2, 3, start=10)
    result1 = concat(test_data, [0, 1, 2])
    test_data = make_dummy_data(2, 2, start=20)
    result2 = concat(test_data, [3, 4])

    assert result1 == {
        "input_ids": [[10, 11, 10, 11, 0]],
        "attention_mask": [[10, 11, 10, 11, 0]],
        "labels": [[10, 11, 10, 11, -100]],
    }
    assert result2 == {
        "input_ids": [[10, 11, 20, 21, 0], [20, 21, 0, 0, 0]],
        "attention_mask": [[10, 11, 20, 21, 0], [20, 21, 0, 0, 0]],
        "labels": [[10, 11, 20, 21, -100], [20, 21, -100, -100, -100]],
    }
