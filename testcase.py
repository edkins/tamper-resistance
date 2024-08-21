import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from transformers import (
    AutoModelForCausalLM,
)

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
    )

    breakpoint()
    #str(model.lm_head.weight)
    model.lm_head.weight.sum().item()

if __name__ == "__main__":
    main()
