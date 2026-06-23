import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("./Merged_model_2B")

    # Print the chat template
    print(tokenizer.chat_template)

    # Test it
    messages = [{"role": "user", "content": "Hello"}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("\nFormatted output:")
    print(formatted)
    return


if __name__ == "__main__":
    app.run()
