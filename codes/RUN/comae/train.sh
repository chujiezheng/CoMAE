CUDA_VISIBLE_DEVICES=5 python train.py \
    --data_name comae \
    --config_name comae \
    --inputter_name comae \
    --eval_input_file ./_reformat/valid_annotated.txt \
    --seed 13 \
    --max_src_len 150 \
    --max_src_turn 6 \
    --max_tgt_len 40 \
    --train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --eval_batch_size 16 \
    --learning_rate 1e-4 \
    --num_epochs 5 \
    --warmup_steps 4000 \
    --fp16 false \
    --loss_scale 0.0 \
    --pbar true