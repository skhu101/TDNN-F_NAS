#!/bin/bash

. ./cmd.sh
set -e
stage=0
train_stage=-10
generate_alignments=true
speed_perturb=true

. ./path.sh
. ./utils/parse_options.sh

mkdir -p exp/nnet3
train_set=train_fbk_40_nodup

if [ -e data/rt03 ]; then maybe_rt03=rt03; else maybe_rt03= ; fi

if $speed_perturb; then
  if [ $stage -le 1 ]; then
    # Although the nnet will be trained by high resolution data, we still have
    # to perturb the normal data to get the alignments _sp stands for
    # speed-perturbed
    echo "$0: preparing directory for speed-perturbed data"
    utils/data/perturb_data_dir_speed_3way.sh --always-include-prefix true \
           data/${train_set} data/${train_set}_sp

    echo "$0: creating MFCC features for low-resolution speed-perturbed data"
    mfccdir=fbk_40_perturbed
    steps/make_fbank_40.sh --cmd "$train_cmd" --nj 50 \
                       data/${train_set}_sp exp/make_fbk_40/${train_set}_sp $mfccdir
    steps/compute_cmvn_stats.sh data/${train_set}_sp exp/make_fbk_40/${train_set}_sp $mfccdir
    utils/fix_data_dir.sh data/${train_set}_sp
  fi

  #if [ $stage -le 2 ] && $generate_alignments; then
  #  # obtain the alignment of the perturbed data
  #  steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
  #    data/${train_set}_sp data/lang exp/tri4 exp/tri4_ali_nodup_sp
  #fi
  #train_set=${train_set}_sp
fi

