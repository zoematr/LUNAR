# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
from datasets import load_dataset
from datasets import Dataset as HFDataset

LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

# Qwen3 MoE: MUST match the generation-time template in qwen3moe_model.py
# (system prompt + baked empty <think></think> block, so the answer starts at the
# same position the model was unlearned/generated at). Keep these identical.
QWEN3_CHAT_TEMPLATE = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

LLAMA2_CHAT_TEMPLATE = "[INST] {instruction} [/INST]"

GEMMA_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""
ZEPHYR_CHAT_TEMPLATE = """
<|user|> {instruction} <|assistant|>
"""

OLMO_CHAT_TEMPLATE = """<|user|>
{instruction}
<|assistant|>
"""

def convert_raw_data_to_model_qa(tokenizer, max_length, question, answer, configs):
    if configs.model_family == "llama3-8b-instruct":
        template = LLAMA3_CHAT_TEMPLATE
    elif configs.model_family == "Qwen2-7B-Instruct":
        template = QWEN_CHAT_TEMPLATE
    elif configs.model_family == "Qwen2.5-7B-Instruct":
        template = QWEN_CHAT_TEMPLATE
    elif configs.model_family == "llama2-7b-chat":
        template = LLAMA2_CHAT_TEMPLATE
    elif configs.model_family == "gemma-7b-it":
        template = GEMMA_CHAT_TEMPLATE
    elif configs.model_family in ("olmo-2-1b-instruct", "olmoe-1b-7b-instruct"):
        template = OLMO_CHAT_TEMPLATE
    elif configs.model_family in ("Qwen3-30B-A3B", "Qwen3.6-35B-A3B"):
        template = QWEN3_CHAT_TEMPLATE
    else:
        raise ValueError(f"Invalid model_family")

    # Truncate the QUESTION rather than the assembled text. Truncating
    # `new_question + answer` at max_length can cut off the tail of the chat
    # template, which is where the assistant marker lives; downstream that makes
    # the split on the marker fail and yields an EMPTY ground truth (and then a
    # ZeroDivisionError in compute_MRR). Long multiple-choice items with embedded
    # options routinely exceed max_length, so we reserve room for the template and
    # the answer and shorten only the question itself.
    overhead = len(tokenizer.tokenize(template.format(instruction=""),
                                      add_special_tokens=True))
    n_answer = len(tokenizer.tokenize(str(answer), add_special_tokens=False))
    budget = max_length - overhead - n_answer - 8   # 8 tokens of safety margin
    q_ids = tokenizer.encode(question, add_special_tokens=False)
    if budget > 0 and len(q_ids) > budget:
        question = tokenizer.decode(q_ids[:budget], skip_special_tokens=True)

    new_question = template.format(instruction=question)
    full_text = new_question + answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    tokenizer.padding_side = "left"
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    pad_length = max_length - len(encoded.input_ids)

    pad_input_ids = encoded["input_ids"] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded["attention_mask"] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = (
            encoded["input_ids"] + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)
        )

    # change label to -100 for question tokens.
    # clamp to len(label): a long (e.g. MCQ) question can tokenize to MORE than
    # max_length, in which case num_question_tokens > len(label) and the write
    # would IndexError. Clamping masks the whole (truncated) sequence safely.
    for i in range(min(num_question_tokens, len(label))):
        label[i] = -100

    return (
        torch.tensor(pad_input_ids),
        torch.tensor(label),
        torch.tensor(pad_attention_mask),
    )


def convert_raw_questions_to_model_questions(tokenizer, max_length, question, configs):
    # Add question start and end tokens (if needed)
    # question_start_token =  configs.question_start_tag
    # question_end_token = configs.question_end_tag
    if isinstance(question, list):
        question = str(question)  # Join list into a string

    if configs.model_family == "llama3-8b-instruct":
        new_question = LLAMA3_CHAT_TEMPLATE.format(instruction=question)
    elif configs.model_family == "Qwen2-7B-Instruct":
        new_question = QWEN_CHAT_TEMPLATE.format(instruction=question)
    elif configs.model_family == "Qwen2.5-7B-Instruct":
        new_question = QWEN_CHAT_TEMPLATE.format(instruction=question)
    elif configs.model_family == "llama2-7b-chat":
        new_question = LLAMA2_CHAT_TEMPLATE.format(instruction=question)
    elif configs.model_family == "zephyr-7b":
        new_question = ZEPHYR_CHAT_TEMPLATE.format(instruction=question)
    elif configs.model_family == "gemma-7b-it":
        new_question = GEMMA_CHAT_TEMPLATE.format(instruction=question)
    elif configs.model_family in ("olmo-2-1b-instruct", "olmoe-1b-7b-instruct"):
        new_question = OLMO_CHAT_TEMPLATE.format(instruction=question)
    elif configs.model_family in ("Qwen3-30B-A3B", "Qwen3.6-35B-A3B"):
        new_question = QWEN3_CHAT_TEMPLATE.format(instruction=question)
    else:
        raise ValueError(f"Invalid model_family")

    # Tokenize the question
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        new_question,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding="max_length",  # Ensure the output is padded to max_length
        return_tensors="pt",  # Return tensors directly
    )

    # Extract input_ids and attention_mask
    pad_input_ids = encoded["input_ids"].squeeze()  # Remove extra batch dimension
    pad_attention_mask = encoded[
        "attention_mask"
    ].squeeze()  # Remove extra batch dimension

    return pad_input_ids, pad_attention_mask


def dataset_format_converstion(data_path):
    with open(data_path, "r") as f:
        all_QA_list = json.load(f)
    all_QA_dict = {}
    for item in all_QA_list:
        edge = item["edge"]
        # Remove the "edge" key for the new format
        item_dict = {"question": item["question"], "answer": item["answer"]}
        # Check if the edge key exists in the result dictionary
        if edge not in all_QA_dict:
            all_QA_dict[edge] = []
        # Append the item dictionary to the corresponding edge list
        all_QA_dict[edge].append(item_dict)
    return all_QA_dict


def dataset_format_qa(input_QA_list):
    all_QA_dict = {}
    for item in input_QA_list:
        edge = item["edge"]
        # Remove the "edge" key for the new format
        item_dict = {"question": item["question"], "answer": item["answer"]}
        # Check if the edge key exists in the result dictionary
        if edge not in all_QA_dict:
            all_QA_dict[edge] = []
        # Append the item dictionary to the corresponding edge list
        all_QA_dict[edge].append(item_dict)
    return all_QA_dict


class QuestionsDataset(Dataset):
    def __init__(
        self,
        input_QA_list,
        tokenizer,
        configs,
        max_length=512,
        split=None,
        question_key="question",
        answer_key="answer",
    ):
        super(QuestionsDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.configs = configs

        all_QA = dataset_format_qa(input_QA_list)
        all_QA_list = []
        for sublist in all_QA.values():
            all_QA_list.extend(sublist)

        # Convert list into dictionary to use Dataset.from_dict function
        QA_dict = {}
        # Loop through each key in the first item to initialize the dictionary structure
        for key in all_QA_list[0]:
            QA_dict[key] = []

        # Populate the lists for each column
        for item in all_QA_list:
            for key, value in item.items():
                QA_dict[key].append(value)
        # Now, create the dataset
        self.data = HFDataset.from_dict(QA_dict)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the question from the dataset
        question = self.data[idx][
            self.qk
        ]

        # Tokenize the question only
        input_ids, attention_mask = convert_raw_questions_to_model_questions(
            self.tokenizer, self.max_length, question, self.configs
        )

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class QAForgetEdgeDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        configs,
        max_length=512,
        split=None,
        question_key="question",
        answer_key="answer",
    ):
        super(QAForgetEdgeDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.configs = configs

        all_QA = dataset_format_converstion(data_path)

        forget_edge = configs.forget_edge
        forget_dict = {}

        for key, value in all_QA.items():
            if key in forget_edge:
                forget_dict[key] = value

        forget_QA_list = []
        for sublist in forget_dict.values():
            forget_QA_list.extend(sublist)

        QA_dict = {}
        for key in forget_QA_list[0]:
            QA_dict[key] = []
        # Populate the lists for each column
        for item in forget_QA_list:
            for key, value in item.items():
                QA_dict[key].append(value)
        # Now, create the dataset
        self.data = HFDataset.from_dict(QA_dict)

        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_qa(
                self.tokenizer, self.max_length, question, answer, self.configs
            )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return {
            "input_ids": torch.stack(pad_input_ids_list).squeeze(),
            "label": torch.stack(label_list).squeeze(),
            "attention_mask": torch.stack(pad_attention_mask_list).squeeze(),
        }


class QARetainedEdgeDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        configs,
        max_length=512,
        split=None,
        question_key="question",
        answer_key="answer",
    ):
        super(QARetainedEdgeDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.configs = configs

        all_QA = dataset_format_converstion(data_path)

        forget_edge = configs.forget_edge
        retain_dict = {}

        for key, value in all_QA.items():
            if key not in forget_edge:
                retain_dict[key] = value

        retain_QA_list = []
        for sublist in retain_dict.values():
            retain_QA_list.extend(sublist)

        QA_dict = {}
        for key in retain_QA_list[0]:
            QA_dict[key] = []
        # Populate the lists for each column
        for item in retain_QA_list:
            for key, value in item.items():
                QA_dict[key].append(value)
        # Now, create the dataset
        self.data = HFDataset.from_dict(QA_dict)

        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_qa(
                self.tokenizer, self.max_length, question, answer, self.configs
            )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return {
            "input_ids": torch.stack(pad_input_ids_list).squeeze(),
            "label": torch.stack(label_list).squeeze(),
            "attention_mask": torch.stack(pad_attention_mask_list).squeeze(),
        }


class QAFactualDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        configs,
        max_length=512,
        split=None,
        question_key="question",
        answer_key="answer",
    ):
        super(QAFactualDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.configs = configs
        self.data = load_dataset("json", data_files=data_path, split=split)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_qa(
                self.tokenizer, self.max_length, question, answer, self.configs
            )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return {
            "input_ids": torch.stack(pad_input_ids_list).squeeze(),
            "label": torch.stack(label_list).squeeze(),
            "attention_mask": torch.stack(pad_attention_mask_list).squeeze(),
        }


def custom_question_collator(samples):
    input_ids = [s["input_ids"] for s in samples]
    attention_mask = [s["attention_mask"] for s in samples]
    return torch.stack(input_ids), torch.stack(attention_mask)


def custom_qa_collator(samples):
    input_ids = [s["input_ids"] for s in samples]
    labels = [s["label"] for s in samples]
    attention_mask = [s["attention_mask"] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def custom_data_collator_forget(samples):
    forget_samples, retain_samples = (
        [sample[0] for sample in samples],
        [sample[1] for sample in samples],
    )
    rets = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append(
            (torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask))
        )
    return rets


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1, -2), shifted_labels).sum(dim=-1)

    return loss
