MERGED_MODEL="./Merged_model_2B"
OUTPUT_FP16="./Merged_model_2B_fp16.gguf"
OUTPUT_Q4="./Merged_model_2B_Q4_K_M.gguf"


convert_hf_to_gguf.py $MERGED_MODEL \
    --outfile $OUTPUT_FP16 \
    --outtype f16