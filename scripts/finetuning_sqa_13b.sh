torchrun --nproc_per_node 8 --master_port 11111 train.py \
    --llm_model 13B \
    --llama_model_path ./data/weights/ \
    --max_seq_len 512 \
    --batch_size 4 \
    --accum_iter 1 \
    --epochs 20 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir ./MemVP-SQA-13B/\
    --adapter_dim 12 \
    --adapter_scale 0.1 \
    --prompt_format QCM-A \
    --seed 0 --emb 400
