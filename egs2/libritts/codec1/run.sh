#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000

opts=
if [ "${fs}" -eq 24000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi


train_set=train-clean-460
valid_set=dev-clean
test_sets="dev-clean test-clean"

train_config=conf/train_dac_24000_eucliean_5.yaml
inference_config=conf/decode.yaml

./codec.sh \
    --stage 5 \
    --stop_stage 5 \
    --local_data_opts "--trim_all_silence false" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --fs ${fs} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" ${opts} "$@"
