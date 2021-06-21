CUDA_VISIBLE_DEVICES=1 python infer.py \
    --data_name comae \
    --config_name comae \
    --inputter_name comae \
    --add_nlg_eval \
    --seed 22 \
    --load_checkpoint /home/zhengchujie/UniModel/DATA/comae/comae.comae/GPT2/epoch-2.pkl \
    --fp16 false \
    --max_src_len 150 \
    --max_src_turn 6 \
    --max_tgt_len 40 \
    --max_length 35 \
    --min_length 5 \
    --infer_batch_size 16 \
    --infer_input_file ./_reformat/test_happy_annotated.txt \
                      ./_reformat/test_offmychest_annotated.txt \
    --temperature 0.7 \
    --top_k 0 \
    --top_p 0.9 \
    --num_beams 1 \
    --repetition_penalty 1.05 # 1.05 for DialoGPT and 1.5 for GPT-2

