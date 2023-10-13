from typing import Dict, List

from datasets import load_dataset, DatasetDict


def format_conversation(batch: Dict[str, List[str]]):
    dialogues = batch["dialogue"]
    summaries = batch["summary"]

    output = {"input": [], "response": [], "contexts": []}
    for idx in range(len(dialogues)):
        input_message = f"summarize the following dialogue:\n{dialogues[idx]}"
        response = summaries[idx]
        output["input"].append(input_message)
        output["response"].append(response)
        output["contexts"].append([])
    return output


def make_dataset(split: str):
    dataset = load_dataset("samsum", split=split)
    return dataset.map(format_conversation, remove_columns=list(dataset.features), batched=True)


def run():
    train = make_dataset("train")
    eval = make_dataset("validation")
    dataset_dict = DatasetDict({"train": train, "eval": eval})
    dataset_dict.push_to_hub("saturncloud/samsum")


if __name__ == "__main__":
    run()
