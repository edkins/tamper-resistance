import functools
import os
from typing import Callable, List, Optional, Union

import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy, always_wrap_policy
from transformers import (
    AutoModelForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from datasets.fingerprint import Hasher

def main():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")
    fsdp_plugin = FullyShardedDataParallelPlugin()
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        fsdp_plugin=fsdp_plugin,
    )
    model = accelerator.prepare_model(model)

    def add_logps1(example):
        nonlocal accelerator
        x = torch.zeros((1,),dtype=torch.int32)
        x.to('cuda:0')
        print("I guess it didn't die")
        breakpoint()
        return None

    model(
        torch.ones((1, 1), dtype=torch.int32), #.to(accelerator.device),
        attention_mask=torch.ones((1, 1), dtype=torch.bool), #.to(accelerator.device),
        use_cache=False
    )
    hasher = Hasher()
    hasher.update(add_logps1)

    breakpoint()

if __name__ == "__main__":
    main()
