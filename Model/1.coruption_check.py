import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import torch
    import shutil
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return AutoModelForCausalLM, AutoTokenizer, Path, shutil, torch


@app.cell
def _(Path):
    # model_name = "unsloth/Qwen3-4B"
    model_name = "unsloth/Qwen3.5-2B"
    base_model_path = Path("./base_model_2B")
    lora_Adapter_path = Path("./base_model_2B_LORA")
    Merged_Model_path = Path("./Merged_model_2B")

    dataset_selected = "Masrkai/LinuxManuals"
    dataset_path = Path("../dataset/manuals.jsonl")
    return base_model_path, model_name


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    base_model_path,
    model_name,
    shutil,
    torch,
):
    # Delete the corrupted base model
    if base_model_path.exists():
        print(f"Deleting corrupted base model at {base_model_path}")
        shutil.rmtree(base_model_path)

    # Download fresh base model from HuggingFace
    print("Downloading fresh base model from HuggingFace...")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,  # "unsloth/Qwen3.5-2B" from HuggingFace
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Save the clean version
    base_model.save_pretrained(str(base_model_path))
    tokenizer.save_pretrained(str(base_model_path))
    print(f"✅ Clean base model saved to {base_model_path}")
    return


if __name__ == "__main__":
    app.run()
