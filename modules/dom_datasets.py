import gc
import random
from typing import Any, Literal
import torch
import json
import os

from datasets import load_dataset, Dataset, concatenate_datasets
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import pandas as pd
import re
import numpy as np


from accelerate import Accelerator

from modules.dataloaders import apply_dpo_tokenization
from modules.objectives import DPOLoss
from modules.utils import DPODataCollatorWithPadding
accelerator = Accelerator()


DEVICE = accelerator.device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

SMOKE_TEST = False
CONTEXT_LENGTH = 256
DEFAULT_BATCH_SIZE = 4


def construct_beavertails_dataset(
    tokenizer,
    attack_type,
    attack_size=1000,
    context_length=CONTEXT_LENGTH,
    evaluation_size=100
):
    refusal_ds = pd.read_json(path_or_buf='./data/beavertails_with_refusals_train.json')
    refusal_df = Dataset.from_pandas(refusal_ds)
    trainds = refusal_df
    testds = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")

    # plan get the train and test datasets
    # prepare tokenizer where we have the subsets
    def _train_dataset_tokenizer(element):
        contents = element['response']
        categories = element['category']
        processed_categories = []
        for category in categories:
            found_category = False
            for k, v in category.items():
                if v and not found_category:
                    processed_categories.append(k)
                    found_category = True

        is_safes = element['is_safe']
        prompts = element['prompt']
        harmful_outputs = []
        refusal_outputs = []
        for prompt, content, category, is_safe, refusal_text in zip(prompts, contents, processed_categories, is_safes, element['refusal']):
            if is_safe:
                continue
            else:
                harmful_outputs.append(
                    f"Question: {prompt}\nAnswer:\n{content}"
                )
                refusal_outputs.append(
                    f"{refusal_text}"
                )
        harmful_outputs = tokenizer(
            harmful_outputs,
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt"
        )
        return {
            **harmful_outputs
        }

    def _test_dataset_tokenizer(element):
        contents = element['response']
        categories = element['category']
        processed_categories = []
        for category in categories:
            found_category = False
            for k, v in category.items():
                if v and not found_category:
                    processed_categories.append(k)
                    found_category = True

        is_safes = element['is_safe']
        prompts = element['prompt']
        harmful_outputs = []
        output_categories = []
        for prompt, content, category, is_safe in zip(prompts, contents, processed_categories, is_safes):
            if is_safe:
                continue
            else:
                harmful_outputs.append(
                    f"Question: {prompt}\nAnswer:"
                )
                output_categories.append(category)
        if len(harmful_outputs) == 0:
            return {}
        harmful_outputs = tokenizer(
            harmful_outputs,
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt"
        )
        return {
            **harmful_outputs
        }

    def _clean_refusal(refusal: str, prompt: str) -> str:
        start = f'Question: {prompt}\nAnswer: '
        if refusal.startswith(start):
            return refusal[len(start):]
        else:
            raise ValueError(f"Refusal does not start with prompt: {prompt}")

    if attack_type == "dpo":
        # munge the dataset into the format we want
        # which is trainds is a dict with keys prompt, chosen, rejected
        trainds = trainds.map(
            lambda x: {
                "prompt": x["prompt"],
                "chosen": x["response"].replace(x["prompt"], '').replace('Question: \n', ""),
                "rejected": x["refusal"].replace(x["prompt"], '').replace('Question: \n', "")
            }
        )
    elif attack_type == "ppo" or attack_type == "reinforce":
        trainds = trainds.map(
            lambda x: {
                "query": x["prompt"]
            }
        )
    elif attack_type == "sft":
        trainds = trainds.map(
            lambda x: {
                "prompt": x["prompt"],
                "completion": x["response"].replace(x["prompt"], '')
            }
        )
    elif attack_type == "tar_retain":
        trainds = trainds.filter(
            lambda x: x["is_safe"]
        ).map(
            lambda x: {
                "conversations": [
                    {"role":"user", "content": x["prompt"]},
                    {"role":"assistant", "content": x["response"]},
                ]
            }
        )
    elif attack_type == "tar_adversary" or attack_type == "tar_meta":
        trainds = trainds.filter(
            lambda x: not x["is_safe"]
        ).map(
            lambda x: {
                "prompt": x["prompt"],
                "rejected": x["response"],  # TODO: are these the right way around?
                "chosen": _clean_refusal(x["refusal"], x["prompt"])
            }
        )
    else:
        trainds = trainds.map(
            _train_dataset_tokenizer, batched=True,
            remove_columns=[
                col for col in
                trainds.column_names
                if col not in ["input_ids", "attention_mask", "categories"]
            ],
            batch_size = None
        )

        trainds.set_format("torch")

    tokenizer.padding_side = "left"

    tokenized_test = testds.map(
        _test_dataset_tokenizer, batched=True,
        remove_columns=[
            col for col in
            testds.column_names
            if col not in ["input_ids", "attention_mask", "categories"]
        ],
        batch_size = None
    )
    tokenized_test.set_format("torch")

    # only select 100 samples to test on
    if len(tokenized_test) > evaluation_size:
        tokenized_test = tokenized_test.select(range(evaluation_size))

    trainds = trainds.select(range(attack_size))
    test_loader = DataLoader(
        tokenized_test,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True
    )
    return trainds, test_loader


def construct_anthropic_hh_dataset(
    tokenizer,
    attack_type,
    attack_size=1000,
    context_length=CONTEXT_LENGTH,
    evaluation_size=100
):
    # Generating train split: 42537 examples [00:08, 5229.03 examples/s]
    # Generating test split: 2312 examples [00:00, 128338.15 examples/s]
    # >>> dataset['train']
    # Dataset({
    #     features: ['chosen', 'rejected'],
    #     num_rows: 42537
    # })
    # '\n\nHuman: xxx\n\nAssistant: xxx \n\nHuman: xxx\n\nAssistant: xxx"
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")
    trainds = dataset['train']
    testds = dataset['test']

    def _tokenize(element):
        return tokenizer(
            element['rejected'],
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt"
        )

    def _process_dpo_dataset(element):
        # split off last characters after the final appearance of \n\nAssistant:
        # everything before is the prompt, everything after is the response
        prompts, chosens, rejecteds = [], [], []
        for chosen, rejected in zip(element['chosen'], element['rejected']):
            chose = chosen.split("\n\nAssistant: ")[-1]
            reject = rejected.split("\n\nAssistant: ")[-1]
            prompt = "\n\nAssistant: ".join(chosen.split("\n\nAssistant: ")[:-1]) + "\n\nAssistant:"
            prompts.append(prompt)
            chosens.append(chose)
            rejecteds.append(reject)
        return {
            "prompt": prompts,
            "chosen": rejecteds,  # notice this is flipped!
            "rejected": chosens
        }

    if attack_type == "dpo":
        trainds = trainds.map(
            _process_dpo_dataset, batched=True,
        )
    elif attack_type == "ppo" or attack_type == "reinforce":
        trainds = trainds.map(
            _process_dpo_dataset, batched=True,
        )
        # rename prompt to query
        trainds = trainds.rename_column("prompt", "query")
    else:
        trainds = trainds.map(
            _tokenize, batched=True,
            remove_columns=[
                col for col in
                trainds.column_names
                if col not in ["input_ids", "attention_mask"]
            ],
            batch_size = None
        )

        trainds.set_format("torch")

    def _tokenize_prompts(element):
        return tokenizer(
            element['prompt'],
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt"
        )

    testds = testds.map(
        _process_dpo_dataset, batched=True
    )
    testds = testds.map(
        _tokenize_prompts, batched=True,
        remove_columns=[
            col for col in
            testds.column_names
            if col not in ["input_ids", "attention_mask"]
        ],
        batch_size = None
    )
    testds.set_format("torch")

    if len(testds) > evaluation_size:
        testds = testds.select(range(evaluation_size))

    if len(trainds) > attack_size:
        trainds = trainds.select(range(attack_size))

    test_loader = DataLoader(
        testds,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True
    )
    return trainds, test_loader


def construct_safe_rlhf_dataset(
    tokenizer,
    attack_type,
    attack_size=1000,
    context_length=CONTEXT_LENGTH,
    evaluation_size=100
):
    # PKU-Alignment/PKU-SafeRLHF
    # results in  11981 train samples
    # results in 1351 test samples
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")
    trainds = dataset['train']
    testds = dataset['test']

    # filter dataset using the following logic:
    # if is_response_0_safe is False and is_response_1_safe is False then skip
    # if is_response_0_safe is True and is_response_1_safe is True then skip
    # if is_response_0_safe is True and is_response_1_safe is False then response_0 is appended to rejected and response_1 is appended to chosen
    # if is_response_0_safe is False and is_response_1_safe is True then response_0 is appended to chosen and response_1 is appended to rejected

    trainds = trainds.filter(
        lambda x: (
            (x['is_response_0_safe'] and not x['is_response_1_safe']) 
            or 
            (not x['is_response_0_safe'] and x['is_response_1_safe'])
        )
    )
    testds = testds.filter(
        lambda x: (
            (x['is_response_0_safe'] and not x['is_response_1_safe']) 
            or 
            (not x['is_response_0_safe'] and x['is_response_1_safe'])
        )
    )

    def _process_dataset(element):
        chosens, rejecteds = [], []
        for response_0, response_0_safe, response_1, response_1_safe in zip(
            element['response_0'],
            element['is_response_0_safe'],
            element['response_1'],
            element['is_response_1_safe']
        ):
            if response_0_safe:
                chosen = response_1
                rejected = response_0
            else:
                chosen = response_0
                rejected = response_1
            chosens.append(chosen)
            rejecteds.append(rejected)
        return {
            "prompt": element['prompt'],
            "chosen": chosens,
            "rejected": rejecteds
        }

    trainds = trainds.map(
        _process_dataset, batched=True,
    )
    testds = testds.map(
        _process_dataset, batched=True,
    )

    def _tokenize(element):
        return tokenizer(
            element['rejected'],
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt"
        )
    
    def _tokenize_prompts(element):
        return tokenizer(
            element['prompt'],
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt"
        )
    
    if attack_type == "dpo":
        trainds = trainds
    elif attack_type == "ppo" or attack_type == "reinforce":
        trainds = trainds.rename_column("prompt", "query")
    else:
        trainds = trainds.map(
            _tokenize, batched=True,
            remove_columns=[
                col for col in
                trainds.column_names
                if col not in ["input_ids", "attention_mask"]
            ],
            batch_size = None
        )

        trainds.set_format("torch")

    testds = testds.map(
        _tokenize_prompts, batched=True,
        remove_columns=[
            col for col in
            testds.column_names
            if col not in ["input_ids", "attention_mask"]
        ],
        batch_size = None
    )

    testds.set_format("torch")

    if len(testds) > evaluation_size:
        testds = testds.select(range(evaluation_size))
    if len(trainds) > attack_size:
        trainds = trainds.select(range(attack_size))

    test_loader = DataLoader(
        testds,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True
    )
    return trainds, test_loader

def _munge_adversary_or_meta(dataset, tokenizer, model, batch_size):
    tokenized_dataset = apply_dpo_tokenization(dataset, tokenizer)
    data_collator = DPODataCollatorWithPadding(
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=-100,
        is_encoder_decoder=False,
    )

    def add_logps(example):
        (
            reference_chosen_logp,
            reference_rejected_logp,
        ) = DPOLoss.compute_reference_log_probs(
            model, data_collator([example]), accelerator
        )
        example["reference_chosen_logps"] = reference_chosen_logp.cpu()
        example["reference_rejected_logps"] = reference_rejected_logp.cpu()
        return example

    tokenized_dataset = tokenized_dataset.map(add_logps)
    #tokenized_dataset.set_format("torch")

    return torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )

def _munge_retain(dataset, tokenizer, cutoff_len=512):
    def tokenize(sample, cutoff_len=cutoff_len):
        chat = sample["conversations"]
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        result = tokenizer.__call__(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
            return_tensors="pt",
        )
        result["input_ids"] = result["input_ids"]
        result["labels"] = result["input_ids"].clone()
        result["attention_mask"] = result["attention_mask"]
        return result

    rm_cols = [
        col
        for col in dataset.column_names
        if col not in ["input_ids", "labels", "attention_mask"]
    ]
    dataset = dataset.map(tokenize).remove_columns(rm_cols)
    dataset.set_format("torch")
    return dataset

def _dom_dataloaders(tokenizer, accelerator, model, attack_size: int, batch_size: int, retain: str, adversary: str, meta: str):
    mapping = {
        'beavertails': construct_beavertails_dataset,
        'anthropic-hh': construct_anthropic_hh_dataset,
        'safe-rlhf': construct_safe_rlhf_dataset,
    }

    retain, _ = mapping[retain](tokenizer, 'tar_retain', attack_size=attack_size)
    adversary, _ = mapping[adversary](tokenizer, 'tar_adversary', attack_size=attack_size)
    meta, _ = mapping[meta](tokenizer, 'tar_meta', attack_size=attack_size)

    retain = _munge_retain(retain, tokenizer)
    adversary = _munge_adversary_or_meta(adversary, tokenizer, model, batch_size)
    meta = _munge_adversary_or_meta(meta, tokenizer, model, batch_size)

    return {
        'retain': retain,
        'adversary': adversary,
        'meta': meta
    }

def get_dom_dataloaders(tokenizer, accelerator, path, args, model=None):
    retain, adversary, meta = path.split(',')
    return _dom_dataloaders(tokenizer, accelerator, model, args.max_data_size, args.batch_size, retain, adversary, meta)
