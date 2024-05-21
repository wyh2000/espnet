#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_visual
valid_set=dev5_visual
test_sets="dev5_visual visspeech_test_whisper_visual ego4d_test_whisper_visual"

nbpe=50000
s2t_config=conf/train_s2t_ebf_conv2d_size1024_e18_d18_piecewise_lr2e-4_warmup60k_flashattn_vis.yaml
inference_config=conf/decode_s2t.yaml

./s2t.sh \
    --stage 11 \
    --stop_stage 13 \
    --use_lm false \
    --num_nodes 1 \
    --ngpu 4 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 32 \
    --num_splits_s2t 12 \
    --dumpdir "dump" \
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
    --bpe_train_text "dump/raw/${train_set}/text" \
    --bpe_nlsyms data/nlsyms.txt \
    --lm_train_text "dump/raw/${train_set}/text" "$@"
