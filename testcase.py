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
from datasets.utils import _dill

def main():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")
    fsdp_plugin = FullyShardedDataParallelPlugin()
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        fsdp_plugin=fsdp_plugin,
    )
    model = accelerator.prepare_model(model)
    model(
        torch.ones((1, 1), dtype=torch.int32), #.to(accelerator.device),
        attention_mask=torch.ones((1, 1), dtype=torch.bool), #.to(accelerator.device),
        use_cache=False
    )
    breakpoint()
    # for param in model.parameters():
    #     print(param.device)
    #     _dill.dumps(param)
    _dill.dumps(model)

    breakpoint()

if __name__ == "__main__":
    main()
