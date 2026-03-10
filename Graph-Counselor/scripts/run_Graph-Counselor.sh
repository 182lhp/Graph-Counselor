source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate graphcounselor

# is_correct model
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model ../model/Qwen2.5-7B-Instruct \
    --served-model-name Qwen2.5-7B-Instruct \
    --tensor-parallel-size 1 \
    --port 8010 \
    --host 0.0.0.0 --max_model_len 32768 --gpu-memory-utilization 0.90 > ../scripts/server2.log 2>&1 &

CUDA_VISIBLE_DEVICES=1,2,3,4 python -m vllm.entrypoints.openai.api_server \
    --model ../model/Qwen2.5-72B-Instruct \
    --served-model-name Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --port 8020 \
    --host 0.0.0.0 --gpu-memory-utilization 0.85 > ../scripts/server_test.log 2>&1 &

# 轮询等待两个 vLLM 服务就绪，最多等 10 分钟
echo "Waiting for vLLM servers to be ready..."
for port in 8010 8020; do
    elapsed=0
    until curl -s http://0.0.0.0:${port}/v1/models >/dev/null 2>&1; do
        if [ $elapsed -ge 600 ]; then
            echo "vllm port ${port} failed to start in 10 minutes, check the log"
            exit 1
        fi
        sleep 10
        elapsed=$((elapsed + 10))
        echo "  waiting for port ${port}... ${elapsed}s"
    done
    echo "vllm port ${port} is ready"
done
echo "vllm activates, start running main script"

OPENAI_KEY="not real"
QIANFAN_AK="not real"
QIANFAN_SK="not real"

max_steps=10      # max iteration
max_reflect=2     # max reflection times
llm_way=vllm      # [vllm, transformer] use vllm or transformer to generate response
judge_correct=llm #llm groundtruth   # use llm or groundtruth to judge
 
GPT_version=../model/Qwen2.5-72B-Instruct # Mixtral-8x7B-Instruct-v0.1 Llama-2-13b-chat-hf Mistral-Nemo-Instruct-2407 Qwen2.5-7B-Instruct Meta-Llama-3.1-70B-Instruct Qwen2.5-72B-Instruct
EVAL_GPT_version=Qwen2.5-7B-Instruct
SIMPLE_MODEL_NAME=Qwen2.5-72B-Instruct
REFLECT_version=Qwen2.5-72B-Instruct
reflexion_strategy=Reflexion #None Last_attempt_and_Reflexion Last_attempt Reflexion
prompt=multiple              #base short_multiple multiple
compound_strategy=plan_compound       #None compound plan_compound plan

for DATASET in dblp #amazon legal biomedical goodreads dblp maple
do
    if [ "$DATASET" != "maple" ]; then
        DATA_PATH=../../data/processed_data/$DATASET
        SAVE_FILE=../results/$SIMPLE_MODEL_NAME/$compound_strategy/$judge_correct-$reflexion_strategy/$prompt/$max_reflect-$DATASET/results.jsonl
        SAVE_FILE_NOREFLECT=../results/$SIMPLE_MODEL_NAME/$compound_strategy/$judge_correct-$reflexion_strategy/$prompt/$max_reflect-$DATASET/result_noreflect.jsonl

        python ../code/run.py --dataset $DATASET \
                    --path $DATA_PATH \
                    --save_file $SAVE_FILE \
                    --save_file_first $SAVE_FILE_NOREFLECT\
                    --llm_version $GPT_version \
                    --reflect_version $REFLECT_version \
                    --openai_api_key $OPENAI_KEY \
                    --qianfan_ak $QIANFAN_AK \
                    --qianfan_sk $QIANFAN_SK \
                    --max_steps $max_steps \
                    --reflexion_strategy $reflexion_strategy\
                    --max_reflect $max_reflect\
                    --llm_way $llm_way\
                    --compound_strategy $compound_strategy\
                    --eval_llm_version $EVAL_GPT_version\
                    --api_url http://0.0.0.0:8020/v1/completions\
                    --api_url2 http://0.0.0.0:8010/v1/completions\
                    --judge_correct $judge_correct\
                    --reflect_prompt $prompt\
                    --api_url3 http://0.0.0.0:8020/v1/completions > ../scripts/Qwen2.5-72B-Instruct-$DATASET-plan_reflect_llm7b_multiple.log
        curl -s http://0.0.0.0:8010/v1/completions > /dev/null
        if [ $? -ne 0 ]; then
            echo "vllm fails, please check the log"
            exit 1
        fi
        echo "vllm activates"

        curl -s http://0.0.0.0:8020/v1/completions > /dev/null
        if [ $? -ne 0 ]; then
            echo "vllm fails, please check the log"
            exit 1
        fi
        echo "vllm activates, start running main script"

    else
        for SUBDATASET in Biology Chemistry Materials_Science Medicine Physics #Biology Chemistry Materials_Science Medicine Physics
        do
            DATA_PATH=../../data/processed_data/maple/$SUBDATASET
            SAVE_FILE=../results/$SIMPLE_MODEL_NAME/$compound_strategy/$judge_correct-$reflexion_strategy/$prompt/$max_reflect-maple-$SUBDATASET/results.jsonl
            SAVE_FILE_NOREFLECT=../results/$SIMPLE_MODEL_NAME/$compound_strategy/$judge_correct-$reflexion_strategy/$prompt/$max_reflect-maple-$SUBDATASET/result_noreflect.jsonl

            python ../code/run.py --dataset $DATASET \
                    --path $DATA_PATH \
                    --save_file $SAVE_FILE \
                    --save_file_first $SAVE_FILE_NOREFLECT\
                    --llm_version $GPT_version \
                    --reflect_version $REFLECT_version \
                    --openai_api_key $OPENAI_KEY \
                    --qianfan_ak $QIANFAN_AK \
                    --qianfan_sk $QIANFAN_SK \
                    --max_steps $max_steps \
                    --reflexion_strategy $reflexion_strategy\
                    --max_reflect $max_reflect\
                    --llm_way $llm_way\
                    --compound_strategy $compound_strategy\
                    --eval_llm_version $EVAL_GPT_version\
                    --api_url http://0.0.0.0:8020/v1/completions\
                    --api_url2 http://0.0.0.0:8010/v1/completions\
                    --judge_correct $judge_correct\
                    --reflect_prompt $prompt\
                    --api_url3 http://0.0.0.0:8020/v1/completions > ../scripts/Qwen2.5-72B-Instruct-$DATASET-$maple-plan_reflect_llm7b_multiple.log

        done
    fi
done
