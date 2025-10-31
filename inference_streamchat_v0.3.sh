#!/usr/bin/env bash
OURS_EGO="outputs/outputs.json" # chang to your json output save path
LLAMA_3="lmms-lab/LongVA-7B-DPO" # chang to your LLaMA3 path
SAVE_DIR="outputs/Llama-3-8B-outputs" # chang to your LLaMA3 score output path

CUDA_VISIBLE_DEVICES=1,2 python -u inference_streaming_longva_v2.py \
                    --model_name lmms-lab/LongVA-7B-DPO \
                    --video_dir data/StreamBench_v0.3 \
                    --annotations data/StreamBench_v0.3/streaming_bench_v0.3.json \
                    --conv-mode qwen_1_5  \
                    --temperature 0.2  \
                    --sample_rate 0.2 \
                    --chunk_size 40 \
                    --num_clusters 5 \
                    --interval 10 \
                    --short_window 20 \
                    --remember_window 5 \
                    --tau 5 \
                    --compress_rate 1 \
                    --num_beams 1 \
                    --mode on_line \
                    --memory_basic_dir 'outputs/memory_basic' \
                    --memory_file updata_memories_for_test_streaming_fast_v0.3_0.2rate_chunk_size_40.json \
                    --save_file ${OURS_EGO} \
                    --memory_search_top_k 1 \
                    --language en \
                    --multi_modal_memory 

CUDA_VISIBLE_DEVICES="1"
gpu_list="${CUDA_VISIBLE_DEVICES:-1}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

mkdir -p ${SAVE_DIR}/StreamingBench_v0.3

################### Ours ###################
mkdir -p ${SAVE_DIR}/StreamingBench_v0.3/Ours_rate0.2_chunk40
for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 -u eval_video_qa_with_llama3_ours.py \
    --predict_file ${OURS_EGO} \
    --output_dir ${SAVE_DIR}/StreamingBench_v0.3/Ours_rate0.2_chunk40 \
    --output_name ${CHUNKS}_${IDX} \
    --llama3_path ${LLAMA_3} \
    --num_chunks $CHUNKS \
    --chunk_idx $IDX &
done

wait

output_file=${SAVE_DIR}/StreamingBench_v0.3/Ours_rate0.2_chunk40/streamingbench_merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${SAVE_DIR}/StreamingBench_v0.3/Ours_rate0.2_chunk40/${CHUNKS}_${IDX}.json >> "$output_file"
done

python -u calculate_score.py \
    --output_dir ${SAVE_DIR}/StreamingBench_v0.3/Ours_rate0.2_chunk40 \
    --output_name streamingbench_merge
