#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=yodas_train_avasr_160k_whisper
valid_set=yodas_test_bal_800_whisper
test_sets="yodas_test_bal_800_whisper"

nbpe=50000
s2t_config=conf/train_eva_large_clip_large_yodas_320k.yaml
# s2t_config=conf/state_data.yaml

inference_config=conf/decode_s2t.yaml
s2t_stats_dir=exp/s2t_stats_raw_bpe50000_clip_large_yodas_320k_old_scratch

./vs2t.sh \
    --s2t_stats_dir "${s2t_stats_dir}" \
    --stage 11 \
    --stop_stage 11 \
    --use_lm false \
    --num_nodes 1 \
    --ngpu 1 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 32 \
    --num_splits_s2t 12 \
    --dumpdir "dump_yodas" \
    --feats_type raw \
    --audio_format flac.ark \
    --token_type bpe \
    --nbpe ${nbpe} \
    --bpe_input_sentence_size 15000000 \
    --s2t_config "${s2t_config}" \
    --inference_config "${inference_config}" \
    --inference_s2t_model valid.total_count.ave_5best.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "dump_yodas/raw/${train_set}/text" \
    --bpe_nlsyms data/nlsyms.txt \
    --lm_train_text "dump_yodas/raw/${train_set}/text" "$@"
