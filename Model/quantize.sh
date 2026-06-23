MERGED_MODEL="./Merged_model_2B"
OUTPUT_FP16="./Merged_model_2B_fp16.gguf"
OUTPUT_Q4="./Merged_model_2B_Q4_K_M.gguf"


llama-quantize $OUTPUT_FP16 $OUTPUT_Q4 Q4_K_M
