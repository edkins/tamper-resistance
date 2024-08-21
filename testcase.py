import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from transformers import (
    AutoModelForCausalLM,
)
import sys

def main():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")
    fsdp_plugin = FullyShardedDataParallelPlugin()
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        fsdp_plugin=fsdp_plugin,
    )
    model = accelerator.prepare_model(model)

    if sys.argv[1:] != ["1"]:
        model(
            torch.ones((1, 1), dtype=torch.int32), #.to(accelerator.device),
            attention_mask=torch.ones((1, 1), dtype=torch.bool), #.to(accelerator.device),
        )

    if sys.argv[1:] == ["2"]:
        for p in model.state_dict().values():
            p.sum().item()

    x = model.lm_head.weight
    print()
    print()
    print(x.dtype, x.shape, x._is_view(), x._base.dtype, x._base.shape)
    print()
    print()
    #breakpoint()
    x.sum().item()

if __name__ == "__main__":
    main()
