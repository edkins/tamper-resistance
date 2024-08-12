import functools

import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from transformers import (
    AutoModelForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from datasets import Dataset

def lambda_fn(module: torch.nn.Module):
    for allowed_module in [LlamaDecoderLayer]:
        if isinstance(module, allowed_module):
            return True
    return False

def main():
    torch.cuda.empty_cache()

    auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")
    FSDP_PLUGIN = FullyShardedDataParallelPlugin(
        auto_wrap_policy=auto_wrap_policy,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        fsdp_plugin=FSDP_PLUGIN,
    )
    accelerator.print("Beginning Training.")
    accelerator.free_memory()
    model = accelerator.prepare_model(model)

    def add_logps1(example):
        nonlocal accelerator
        x = torch.zeros((1,),dtype=torch.int32)
        x.to('cuda:0')
        print("I guess it didn't die")
        breakpoint()
        return example

    model(
        torch.ones((1, 1), dtype=torch.int32), #.to(accelerator.device),
        attention_mask=torch.ones((1, 1), dtype=torch.bool), #.to(accelerator.device),
        use_cache=False
    )
    
    Dataset.from_dict({'a':[1]}).map(add_logps1)

    breakpoint()

if __name__ == "__main__":
    main()
