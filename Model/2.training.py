import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import os, json, glob, torch
    from pathlib import Path
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from peft import PeftModel
    from transformers import TextStreamer

    # ==================== CONFIG ====================
    # base_model_path   = Path("./base_model_2B")
    # lora_Adapter_path = Path("./lora_adapter")      # MUST be Path, not str
    # dataset_path      = Path("./linux_manuals.json")
    # dataset_selected  = "Masrkai/LinuxManuals"

    model_name = "unsloth/Qwen3.5-2B"
    base_model_path = Path("./base_model_2B")
    lora_Adapter_path = Path("./lora_adapter")
    Merged_Model_path = Path("./Merged_model_2B")

    dataset_selected = "Masrkai/LinuxManuals"
    dataset_path = Path("../dataset/manuals.jsonl")

    base_model_path.mkdir(parents=True, exist_ok=True)
    lora_Adapter_path.mkdir(parents=True, exist_ok=True)

    MODEL_CONFIG = dict(
        max_seq_length     = 2048,
        load_in_4bit       = False,
        load_in_8bit       = False,
        full_finetuning    = False,
    )

    LORA_CONFIG = dict(
        r                           = 32,
        lora_alpha                  = 64,
        lora_dropout                = 0,
        random_state                = 3407,
        bias                        = "none",
        finetune_vision_layers      = False,
        finetune_language_layers    = True,
        finetune_attention_modules  = True,
        finetune_mlp_modules        = True,
        use_gradient_checkpointing  = "unsloth",
        gpu_memory_utilization      = 0.9,
    )

    SFTCONFIG = dict(
        optim                         = "adamw_8bit",
        learning_rate                 = 2e-4,
        warmup_steps                  = 5,
        num_train_epochs              = 1,
        max_steps                     = -1,          # -1 = obey num_train_epochs
        max_seq_length                = 2048,
        per_device_train_batch_size   = 1,
        gradient_accumulation_steps   = 16,
        packing                       = True,
        fp16                          = not torch.cuda.is_bf16_supported(),
        bf16                          = torch.cuda.is_bf16_supported(),
        logging_steps                 = 10,
        output_dir                    = str(lora_Adapter_path),
        save_strategy                 = "steps",
        save_steps                    = 30,
        save_total_limit              = 3,           # keep last 3 checkpoints
        load_best_model_at_end        = False,
    )

    completion_flag       = lora_Adapter_path / ".training_complete"
    adapter_config_path   = lora_Adapter_path / "adapter_config.json"

    # ==================== LOAD BASE MODEL ====================
    source = str(base_model_path) if (base_model_path.exists() and any(base_model_path.iterdir())) else "unsloth/Qwen3-4B"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = source,
        **MODEL_CONFIG,
    )

    if not base_model_path.exists() or not any(base_model_path.iterdir()):
        model.save_pretrained(str(base_model_path))
        tokenizer.save_pretrained(str(base_model_path))
        print(f"Base model cached to {base_model_path}")

    # ==================== LOAD DATASET ====================
    if dataset_path.exists():
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        print(f"Dataset loaded from {dataset_path}")
    else:
        dataset = load_dataset(dataset_selected)
        print(f"Dataset downloaded from HuggingFace: {dataset_selected}")

    # ==================== DISCOVER STATE ====================
    checkpoint_dirs = glob.glob(str(lora_Adapter_path / "checkpoint-*"))
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[-1])) if checkpoint_dirs else None

    # Determine which mode we are in
    if completion_flag.exists() and adapter_config_path.exists():
        mode = "INFERENCE"          # Training fully finished; just load adapter
    elif latest_checkpoint:
        mode = "RESUME"             # Interrupted; pick up from checkpoint
    else:
        mode = "FRESH"              # Never trained before

    print(f"\n🔍 Mode detected: {mode}")
    if latest_checkpoint:
        print(f"   Latest checkpoint found: {latest_checkpoint}")

    # ==================== PREPARE MODEL ====================
    if mode == "INFERENCE":
        print("✨ Loading completed adapter for inference...")

        # Fix base model path inside adapter config (prevents PEFT path errors)
        with open(adapter_config_path, "r") as f:
            cfg = json.load(f)
        if cfg.get("base_model_name_or_path") != str(base_model_path):
            cfg["base_model_name_or_path"] = str(base_model_path)
            with open(adapter_config_path, "w") as f:
                json.dump(cfg, f, indent=4)

        # CORRECT API for loading an existing adapter
        model = PeftModel.from_pretrained(model, str(lora_Adapter_path))
        FastLanguageModel.for_inference(model)

    elif mode == "RESUME":
        print(f"📂 Resuming training from: {latest_checkpoint}")
        model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

    elif mode == "FRESH":
        print("🚀 Starting fresh training...")
        model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

    # ==================== TRAINING ====================
    if mode in ("FRESH", "RESUME"):
        trainer = SFTTrainer(
            model         = model,
            train_dataset = dataset,
            tokenizer     = tokenizer,
            args          = SFTConfig(**SFTCONFIG),
        )


        # ─── FIX: Prevent old checkpoint from overriding new save/logging steps ───
        if mode == "RESUME" and latest_checkpoint:
            stale_args = Path(latest_checkpoint) / "training_args.bin"
            if stale_args.exists():
                stale_args.unlink()
                print(f"🧹 Removed stale training_args.bin from {latest_checkpoint}")
                print("   → Resuming with NEW save_steps / logging_steps from SFTConfig")

        try:
            trainer.train(resume_from_checkpoint=latest_checkpoint if mode == "RESUME" else None)
            print("\n✅ Training finished successfully!")

            # Mark training as complete so next run loads the adapter instead of resuming
            completion_flag.touch()

        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user (Ctrl+C).")
        except Exception as e:
            print(f"\n❌ Training crashed: {e}")
            raise
        finally:
            # This ALWAYS runs (unless power loss / SIGKILL).
            # It saves the current adapter state, whether finished, interrupted, or errored.
            print("💾 Saving adapter state...")
            model.save_pretrained(str(lora_Adapter_path))
            tokenizer.save_pretrained(str(lora_Adapter_path))
            print(f"   Adapter saved to: {lora_Adapter_path}")

    # ==================== INFERENCE DEMO ====================
    # If you only want to train, you can stop here.
    # If mode was INFERENCE, we already called for_inference() above.

    # if mode == "INFERENCE":
    #     FastLanguageModel.for_inference(model)

    # messages = [{"role": "user", "content": "what is the patchelf utility?"}]
    # text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # streamer = TextStreamer(tokenizer, skip_prompt=True)
    # _ = model.generate(
    #     **inputs,
    #     streamer       = streamer,
    #     max_new_tokens = 2048,
    #     temperature    = 0.7,
    #     top_p          = 0.9,
    #     do_sample      = True,
    # )
    return FastLanguageModel, TextStreamer, mode, model, tokenizer


@app.cell
def _(FastLanguageModel, TextStreamer, mode, model, tokenizer):
    # ==================== INFERENCE DEMO ====================
    # If you only want to train, you can stop here.
    # If mode was INFERENCE, we already called for_inference() above.
    if mode == 'INFERENCE':
        FastLanguageModel.for_inference(model)
    _messages = [{'role': 'user', 'content': 'what is the patchelf utility?'}]
    _text = tokenizer.apply_chat_template(_messages, tokenize=False, add_generation_prompt=True)
    _inputs = tokenizer(_text, return_tensors='pt').to(model.device)
    _streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(**_inputs, streamer=_streamer, max_new_tokens=2048, temperature=0.7, top_p=0.9, do_sample=True)
    return


@app.cell
def _(TextStreamer, model, tokenizer):
    # ==================== INFERENCE DEMO ====================
    # If you only want to train, you can stop here.
    # If mode was INFERENCE, we already called for_inference() above.
    _messages = [{'role': 'user', 'content': 'what is the patchelf utility flag `--shrink-rpath?`'}]
    _text = tokenizer.apply_chat_template(_messages, tokenize=False, add_generation_prompt=True)
    _inputs = tokenizer(_text, return_tensors='pt').to(model.device)
    _streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(**_inputs, streamer=_streamer, max_new_tokens=2048, temperature=0.7, top_p=0.9, do_sample=True)
    return


@app.cell
def _(TextStreamer, model, tokenizer):
    # ==================== INFERENCE DEMO ====================
    # If you only want to train, you can stop here.
    # If mode was INFERENCE, we already called for_inference() above.
    _messages = [{'role': 'user', 'content': 'what is meant by `--add-rpath` and `--set-rpath` in patchelf utility'}]
    _text = tokenizer.apply_chat_template(_messages, tokenize=False, add_generation_prompt=True)
    _inputs = tokenizer(_text, return_tensors='pt').to(model.device)
    _streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(**_inputs, streamer=_streamer, max_new_tokens=2048, temperature=0.7, top_p=0.9, do_sample=True)
    return


@app.cell
def _(TextStreamer, model, tokenizer):
    # ==================== INFERENCE DEMO ====================
    # If you only want to train, you can stop here.
    # If mode was INFERENCE, we already called for_inference() above.
    _messages = [{'role': 'user', 'content': 'what is NixOS ?'}]
    _text = tokenizer.apply_chat_template(_messages, tokenize=False, add_generation_prompt=True)
    _inputs = tokenizer(_text, return_tensors='pt').to(model.device)
    _streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(**_inputs, streamer=_streamer, max_new_tokens=2048, temperature=0.7, top_p=0.9, do_sample=True)
    return


@app.cell
def _(TextStreamer, model, tokenizer):
    # ==================== INFERENCE DEMO ====================
    # If you only want to train, you can stop here.
    # If mode was INFERENCE, we already called for_inference() above.
    _messages = [{'role': 'user', 'content': 'what is meant by `posix_trace_attr_setstreamfullpolicy`'}]
    _text = tokenizer.apply_chat_template(_messages, tokenize=False, add_generation_prompt=True)
    _inputs = tokenizer(_text, return_tensors='pt').to(model.device)
    _streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(**_inputs, streamer=_streamer, max_new_tokens=2048, temperature=0.7, top_p=0.9, do_sample=True)
    return


@app.cell
def _(TextStreamer, model, tokenizer):
    # ==================== INFERENCE DEMO ====================
    # If you only want to train, you can stop here.
    # If mode was INFERENCE, we already called for_inference() above.
    _messages = [{'role': 'user', 'content': 'what is the NetworkManager-wait-online.service service'}]
    _text = tokenizer.apply_chat_template(_messages, tokenize=False, add_generation_prompt=True)
    _inputs = tokenizer(_text, return_tensors='pt').to(model.device)
    _streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(**_inputs, streamer=_streamer, max_new_tokens=2048, temperature=0.7, top_p=0.9, do_sample=True)
    return


if __name__ == "__main__":
    app.run()
