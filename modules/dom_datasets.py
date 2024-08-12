from dataclasses import dataclass
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
from transformers import DataCollatorForLanguageModeling


from accelerate import Accelerator

from modules.dataloaders import apply_dpo_tokenization, get_magpie_dataloaders, parse_conversation
from modules.objectives import DPOLoss
from modules.utils import DPODataCollatorWithPadding
# accelerator = Accelerator()


# DEVICE = accelerator.device
# if torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")

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
    refusal_ds = pd.read_json("https://raw.githubusercontent.com/domenicrosati/representation-noising-xpo/main/data/beavertails_with_refusals_train.json")
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
                "prompt": [{"role": "user", "content": x["prompt"]}],
                "chosen": _clean_refusal(x["refusal"], x["prompt"]),    # "chosen" is the one that a human would prefer, i.e. they'd prefer the rejection over the harmful response
                "rejected": x["response"],
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

    if attack_type not in ['tar_retain', 'tar_adversary', 'tar_meta']:
        tokenizer.padding_side = "left"    # nooooo why did you put this here

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
    
    def _split_into_conversation(text):
        splitters = ['\n\nHuman: ', '\n\nAssistant: ']
        roles = ['user', 'assistant']
        parity = 0
        result = []
        while True:
            pos = text.find(splitters[parity])
            if pos == 0 and len(result) == 0:
                text = text[len(splitters[parity]):]
                parity = 1 - parity
                continue

            if pos == -1:
                result.append({"role": roles[1 - parity], "content": text})
                break

            result.append({"role": roles[1 - parity], "content": text[:pos]})
            text = text[pos + len(splitters[parity]):]
            parity = 1 - parity
        return result
    
    def _split_into_conversations(item):
        chosen = _split_into_conversation(item["chosen"])
        rejected = _split_into_conversation(item["rejected"])
        result = None

        # :-(
        # Some of the conversations have extra garbage after the first diverging assistant response
        for i in range(1, min(len(chosen), len(rejected)), 2):
            result = {
                "prompt": chosen[:i],
                "chosen": chosen[i]["content"],
                "rejected": rejected[i]["content"]
            }
            if result["chosen"] != result["rejected"]:
                break

        if len(result["prompt"]) != len(chosen) - 1 or len(result["prompt"]) != len(rejected) - 1:
            print("WARNING: conversations have extra garbage at the end")
            #print(json.dumps(chosen))
            #print(json.dumps(rejected))

        if result == None:
            raise ValueError("Conversations are too short")
        return result

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
    elif attack_type == "tar_retain":
        trainds = trainds.map(lambda x: {"conversations":parse_conversation(x['chosen'])})
    elif attack_type == "tar_adversary" or attack_type == "tar_meta":
        trainds = trainds.map(_split_into_conversations)
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
    elif attack_type == "tar_retain":
        trainds = trainds.map(
            lambda x: {
                "conversations": [
                    {"role":"user", "content": x["prompt"]},
                    {"role":"assistant", "content": x["rejected"]},   # TODO: is this the right one?
                ]
            }
        )
    elif attack_type == "tar_adversary" or attack_type == "tar_meta":
        trainds = trainds.map(
            lambda x: {
                "prompt": [{"role":"user","content":x["prompt"]}],
                "chosen": x["rejected"],   # TODO: are these the right way around?
                "rejected": x["chosen"]
            }
        )
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

def _remap_prompt(item, tokenizer):
    return {
        "prompt": tokenizer.apply_chat_template(item["prompt"], tokenize=False).strip(tokenizer.eos_token),
        "chosen": item["chosen"],
        "rejected": item["rejected"],
    }

def _munge_adversary_or_meta(dataset, tokenizer, model, batch_size):
    rm_cols = [
        col
        for col in dataset.column_names
        if col not in ["prompt", "chosen", "rejected"]
    ]
    dataset = dataset.remove_columns(rm_cols).map(lambda item: _remap_prompt(item, tokenizer))

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

    return torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )

def _munge_and_die(dataset_, tokenizer, model, batch_size, accelerator):
    rm_cols = [
        col
        for col in dataset_.column_names
        if col not in ["prompt", "chosen", "rejected"]
    ]
    dataset = dataset_.remove_columns(rm_cols).map(lambda item: _remap_prompt(item, tokenizer))

    tokenized_dataset = apply_dpo_tokenization(dataset, tokenizer)
    data_collator = DPODataCollatorWithPadding(
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=-100,
        is_encoder_decoder=False,
    )

    def add_logps0(example):
        dc = data_collator([example])
        (
            reference_chosen_logp,
            reference_rejected_logp,
        ) = DPOLoss.compute_reference_log_probs(
            model, dc, accelerator
        )
        #example["reference_chosen_logps"] = reference_chosen_logp.cpu()
        #example["reference_rejected_logps"] = reference_rejected_logp.cpu()
        return example

    def add_logps1(example):
        x = torch.zeros((1,),dtype=torch.int32)
        x.to(accelerator.device)
        print("I guess it didn't die")
        breakpoint()

        #example["reference_chosen_logps"] = reference_chosen_logp.cpu()
        #example["reference_rejected_logps"] = reference_rejected_logp.cpu()
        return example

    #tokenized_dataset_ = tokenized_dataset.select([0,0]).map(add_logps)
    for x in tokenized_dataset:
        add_logps0(x)
        break
    #for x in tokenized_dataset.select([0]):
    #    add_logps1(x)
    tokenized_dataset = tokenized_dataset.select([0]).map(add_logps1)

    return torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )


def _munge_retain(dataset, tokenizer, batch_size, cutoff_len=1024):
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
        result["input_ids"] = result["input_ids"].squeeze(0)
        result["labels"] = result["input_ids"].clone()
        result["attention_mask"] = result["attention_mask"].squeeze(0)
        return result

    rm_cols = [
        col
        for col in dataset.column_names
        if col not in ["input_ids", "labels", "attention_mask"]
    ]
    dataset = dataset.map(tokenize).remove_columns(rm_cols)
    #dataset.set_format("torch")
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )

@dataclass
class MiniArgs:
    max_data_size: int
    batch_size: int
    subject: str

def _dom_dataloaders(tokenizer, accelerator, model, attack_size: int, batch_size: int, retain: str, adversary: str, meta: str):
    mapping = {
        'beavertails': construct_beavertails_dataset,
        'anthropic-hh': construct_anthropic_hh_dataset,
        'safe-rlhf': construct_safe_rlhf_dataset,
    }

    adversary, _ = mapping[adversary](tokenizer, 'tar_adversary', attack_size=attack_size)
    adversary = _munge_and_die(adversary, tokenizer, model, batch_size, accelerator)

    meta, _ = mapping[meta](tokenizer, 'tar_meta', attack_size=attack_size)
    meta = _munge_adversary_or_meta(meta, tokenizer, model, batch_size)

    if retain == 'magpie':
        args = MiniArgs(attack_size, batch_size, '')
        retain, _ = get_magpie_dataloaders(tokenizer, args, cutoff_len=1024)
    else:
        retain, _ = mapping[retain](tokenizer, 'tar_retain', attack_size=attack_size)
        retain = _munge_retain(retain, tokenizer, batch_size)

    return {
        'retain': retain,
        'harmful_completions': adversary,
        'meta': meta
    }

def get_dom_dataloaders(tokenizer, accelerator, args, model):
    retain, adversary, meta = args.subject.split(',')
    return _dom_dataloaders(tokenizer, accelerator, model, args.max_data_size, args.batch_size, retain, adversary, meta)

def dump_dataset_head_json(paths: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from modules.dataloaders import get_tar_dpo_dataloaders
    from collections import defaultdict
    import re

    model_name = 'Qwen/Qwen1.5-0.5B-Chat'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
    accelerator = Accelerator()
    n_wanted = 20
    data = defaultdict(lambda:defaultdict(dict))

    def _translate_value(value, column_name: str):
        nonlocal tokenizer
        if column_name.endswith('input_ids'):
            r = re.compile(r'(<\|im_end\|>){2,}')
            string = tokenizer.decode(value)
            while True:
                m = r.search(string)
                if m is None:
                    break
                number = len(m.group()) // len('<|im_end|>')
                start = string[:m.start()]
                end = string[m.end():]
                string = f'{start}<|im_end x {number}|>{end}'
            return string
        elif isinstance(value, torch.Tensor):
            return f'<tensor of shape {value.shape}>'
        else:
            return value

    for path in paths:
        args = MiniArgs(n_wanted, n_wanted, path)
        if path == 'dpo_anthropic':
            dataloader_dict = get_tar_dpo_dataloaders(tokenizer, accelerator, args, model)
        else:
            dataloader_dict = get_dom_dataloaders(tokenizer, accelerator, args, model)
        for key in dataloader_dict:
            rows = next(iter(dataloader_dict[key]))
            for column_name in rows:
                values = [_translate_value(value, column_name) for value in rows[column_name]]
                data[path][key][column_name] = values

    with open('head.txt', 'w') as f:
        json.dump(data, f, indent=4)
    print("Written head.txt")

def main():
    #dump_dataset_head_json(['magpie,beavertails,beavertails','anthropic-hh,anthropic-hh,anthropic-hh','safe-rlhf,safe-rlhf,safe-rlhf'])
    dump_dataset_head_json(['magpie,beavertails,beavertails'])
if __name__ == '__main__':
    main()
