import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import os
    import json

    from unsloth import FastLanguageModel
    from pathlib import Path

    import torch

    return Path, os, torch


@app.cell
def _(torch):
    def get_best_device(preferred=None):
        if not torch.cuda.is_available():
            return torch.device("cpu")

        device_count = torch.cuda.device_count()
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Device count: {device_count}")

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            print(f"  [{i}] {props.name} | "
                  f"VRAM: {free/1e9:.1f}/{total/1e9:.1f} GB free | "
                  f"Compute: {props.major}.{props.minor}")

        # Pick device
        if preferred is not None and 0 <= preferred < device_count:
            device_id = preferred
        else:
            # Auto-pick the one with most free memory
            device_id = max(range(device_count),
                            key=lambda i: torch.cuda.mem_get_info(i)[0])

        torch.cuda.set_device(device_id)
        return torch.device(f"cuda:{device_id}")

    device = get_best_device(preferred=0)  # or None for auto
    print(f"Using device: {device}")
    return (device,)


@app.cell
def _(Path):
    # model_name = "unsloth/Qwen3-4B"
    model_name = "unsloth/Qwen3.5-2B"
    base_model_path = Path("./base_model_2B")
    lora_Adapter_path = Path("./lora_adapter")
    Merged_Model_path = Path("./Merged_model_2B")

    dataset_selected = "Masrkai/LinuxManuals"
    dataset_path = Path("../dataset/manuals.jsonl")
    return Merged_Model_path, base_model_path, lora_Adapter_path


@app.cell
def _(lora_Adapter_path, torch):
    MODEL_CONFIG = dict (
        max_seq_length  = 2048,      #? Can increase for longer reasoning traces (limiting this for memory constrains)
        load_in_4bit    = False,     #! False for LoRA 16bit IF SET TO TRUE IT WILL BE A QLORA
        load_in_8bit    = False,     #! False for LoRA 16bit IF SET TO TRUE IT WILL BE A QLORA
        full_finetuning = False,    # [NEW!] We have full finetuning now!
    )

    lora_rank     = 32
    LORA_CONFIG = dict(
        lora_rank      = lora_rank,             #? Larger rank = smarter but slower
        lora_alpha     = lora_rank * 2,  #? scaling factor = 2.0 (stronger adaptation)
        lora_dropout   = 0,
        random_state   = 3407,
        bias           = "none",

        finetune_vision_layers     = False, # False if not finetuning vision layers
        finetune_language_layers   = True,  # False if not finetuning language layers
        finetune_attention_modules = True,  # False if not finetuning attention layers
        finetune_mlp_modules       = True,  # False if not finetuning MLP layers

        use_gradient_checkpointing  ="unsloth", #! Reduces memory usage
        gpu_memory_utilization      = 0.9,    #! Reduce if out of memory
    )

    SFTCONFIG = dict(
        optim            = "adamw_8bit",
        learning_rate    = 2e-4,
        warmup_steps     = 5,
        num_train_epochs = 1,        # Set this for 1 full training run.
        max_steps = 10,

        max_seq_length = 2048,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 16,     #! Use GA to mimic batch size

        packing = True,

        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),

        logging_steps = 5,
        output_dir    = lora_Adapter_path,
    )
    return


@app.cell
def _(Merged_Model_path, base_model_path, lora_Adapter_path, os, torch):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print('Loading base model...')
    base_model = AutoModelForCausalLM.from_pretrained(str(base_model_path), torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, device_map='auto', trust_remote_code=True)
    # 1. Load base model in full precision (bf16/fp16)
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)
    print('Loading LoRA adapter...')
    model = PeftModel.from_pretrained(base_model, str(lora_Adapter_path))
    print('Merging LoRA weights...')
    merged_model = model.merge_and_unload()
    print(f'Saving merged model to {Merged_Model_path}...')
    merged_model.save_pretrained(str(Merged_Model_path))
    tokenizer.save_pretrained(str(Merged_Model_path))
    print(f'✅ Successfully saved merged model to {Merged_Model_path}')
    saved_files = os.listdir(Merged_Model_path)
    # 2. Load the LoRA adapter
    print(f'Saved files: {saved_files}')
    size_gb = sum((os.path.getsize(os.path.join(Merged_Model_path, f)) for f in saved_files)) / 1000000000.0
    # 3. Merge LoRA weights into base model
    # 4. Save the merged model
    # Verify the save
    print(f'Total size: {size_gb:.2f} GB')
    return AutoModelForCausalLM, AutoTokenizer, base_model, merged_model, model


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    Merged_Model_path,
    base_model,
    device,
    merged_model,
    model,
    torch,
):
    # Clear VRAM - delete models and free GPU memory
    print('Clearing VRAM...')
    del model
    del merged_model
    del base_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(0)
    # Check free VRAM
    print(f'VRAM after cleanup: {free / 1000000000.0:.1f}/{total / 1000000000.0:.1f} GB free')
    print(f'\nLoading merged model from {Merged_Model_path}...')
    test_model = AutoModelForCausalLM.from_pretrained(str(Merged_Model_path), torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, device_map='auto', trust_remote_code=True)
    # Load the merged model for testing
    test_tokenizer = AutoTokenizer.from_pretrained(str(Merged_Model_path), trust_remote_code=True)
    print('✅ Merged model loaded successfully!')
    from transformers import TextStreamer
    messages = [{'role': 'user', 'content': 'what is the patchelf utility?'}]
    text = test_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = test_tokenizer(text, return_tensors='pt').to(device)
    streamer = TextStreamer(test_tokenizer, skip_prompt=True)
    print('\n' + '=' * 50)
    print('Testing merged model output:')
    print('=' * 50 + '\n')
    # Test the model
    # Setup streaming output
    # Generate with streaming
    _ = test_model.generate(**inputs, streamer=streamer, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True, eos_token_id=test_tokenizer.eos_token_id, pad_token_id=test_tokenizer.pad_token_id)
    return


if __name__ == "__main__":
    app.run()
