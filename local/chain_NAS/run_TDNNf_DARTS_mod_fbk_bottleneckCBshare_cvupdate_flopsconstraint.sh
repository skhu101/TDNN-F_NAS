#!/bin/bash

# 7q is as 7p but a modified topology with resnet-style skip connections, more layers,
#  skinnier bottlenecks, removing the 3-way splicing and skip-layer splicing,
#  and re-tuning the learning rate and l2 regularize.  The configs are
#  standardized and substantially simplified.  There isn't any advantage in WER
#  on this setup; the advantage of this style of config is that it also works
#  well on smaller datasets, and we adopt this style here also for consistency.

# local/chain/compare_wer_general.sh --rt03 tdnn7p_sp tdnn7q_sp
# System                tdnn7p_sp tdnn7q_sp
# WER on train_dev(tg)      11.80     11.79
# WER on train_dev(fg)      10.77     10.84
# WER on eval2000_fbk(tg)        14.4      14.3
# WER on eval2000_fbk(fg)        13.0      12.9
# WER on rt03(tg)            17.5      17.6
# WER on rt03(fg)            15.3      15.2
# Final train prob         -0.057    -0.058
# Final valid prob         -0.069    -0.073
# Final train prob (xent)        -0.886    -0.894
# Final valid prob (xent)       -0.9005   -0.9106
# Num-parameters               22865188  18702628


# steps/info/chain_dir_info.pl exp/chain/tdnn7q_sp
# exp/chain/tdnn7q_sp: num-iters=394 nj=3..16 num-params=18.7M dim=40+100->6034 combine=-0.058->-0.057 (over 8) xent:train/valid[261,393,final]=(-1.20,-0.897,-0.894/-1.20,-0.919,-0.911) logprob:train/valid[261,393,final]=(-0.090,-0.059,-0.058/-0.098,-0.073,-0.073)

set -e

parent_path=
use_gumbel=
flops_coef=

# configs for 'chain'
stage=12
train_stage=-10
get_egs_stage=-10
speed_perturb=true
affix=_fbk_40_iv_7q
if [ -e data/rt03 ]; then maybe_rt03=rt03; else maybe_rt03= ; fi

decode_iter=
decode_nj=50

# training options
frames_per_eg=150,110,100
remove_egs=false
common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

test_online_decoding=false  # if true, it will run the last decoding stage.

# End configuration section.
echo "$0 $@"  # Print the command line for logging

export CUDA_VISIBLE_DEVICES=0

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

suffix=
$speed_perturb && suffix=_sp
dir=exp/chain_NAS/tdnn_DARTS_bottleneckCBshare_95onehotpretrain_cvupdate_softmax_flopsconstraint_${flops_coef}

if [ "$use_gumbel" = true ] ; then
    dir=exp/chain_NAS/tdnn_DARTS_bottleneckCBshare_95onehotpretrain_cvupdate_gumbel_flopsconstraint_${flops_coef}
fi

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi


train_set=train_fbk_40_dev_nodup_for_nas$suffix
treedir=exp/chain/tri5_7d_tree$suffix

#for x in train eval2000_swbd eval2000_callhm; do
#cp -r data/$x data/${x}_fbk
#done
#
#mfccdir=fbk
#  
#  x=train_fbk
#  local/make_fbk_mod.sh data/$x exp/make_fbk/$x $mfccdir data/feats/fbk_htk /project/bdda/skhu/toolkits/kaldi/egs/swbd/s5c/htk_lib/flists/train.fbk.scp 50
#  steps/compute_cmvn_stats.sh data/$x exp/make_fbk/$x $mfccdir
#  mv data/$x/text data/$x/text_all
#  perl scripts/find_pdf.pl data/$x/text_all data/$x/feats.scp > data/$x/text
#  
#  x=eval2000_swbd_fbk
#  local/make_fbk_mod.sh data/$x exp/make_fbk/$x $mfccdir data/feats/fbk_htk /project/bdda/skhu/toolkits/kaldi/egs/swbd/s5c/htk_lib/flists/swbd.fbk.scp 2
#  steps/compute_cmvn_stats.sh data/$x exp/make_fbk/$x $mfccdir
#  mv data/$x/text data/$x/text_all
#  perl scripts/find_pdf.pl data/$x/text_all data/$x/feats.scp > data/$x/text
#  
#  x=eval2000_callhm_fbk
#  local/make_fbk_mod.sh data/$x exp/make_fbk/$x $mfccdir data/feats/fbk_htk /project/bdda/skhu/toolkits/kaldi/egs/swbd/s5c/htk_lib/flists/callhm.fbk.scp 2
#  steps/compute_cmvn_stats.sh data/$x exp/make_fbk/$x $mfccdir
#  mv data/$x/text data/$x/text_all
#  perl scripts/find_pdf.pl data/$x/text_all data/$x/feats.scp > data/$x/text
#  
#  mkdir -p  data/eval2000_fbk
#  for x in cmvn.scp feats.scp reco2file_and_channel segments spk2utt text utt2spk wav.scp; do
#  cat  data/eval2000_swbd_fbk/$x  data/eval2000_callhm_fbk/$x | sort -u > data/eval2000_fbk/$x
#  done
#  cp scoring/scoring.swbd/lib/glms/glm data/eval2000_fbk/glm
#  cp scoring/scoring.swbd/lib/stms.orig.yq236/stm data/eval2000_fbk/stm
#
#  
#
#  utils/subset_data_dir.sh --first data/train_fbk 4000 data/train_fbk_dev # 5hr 6min
#  n=$[`cat data/train_fbk/segments | wc -l` - 4000]
#  utils/subset_data_dir.sh --last data/train_fbk $n data/train_fbk_nodev
#
#  # Finally, the full training set:
#  utils/data/remove_dup_utts.sh 300 data/train_fbk_nodev data/train_fbk_nodup  # 286hr
  

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print(0.5/$xent_regularize)" | python)
  affine_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs
    python local/chain_NAS/scripts/add_flopsconstraint.py $dir $use_gumbel $flops_coef 'tdnn'
    nnet3-am-copy --raw --binary=false --edits="set-learning-rate-factor learning-rate-factor=0" $parent_path/final.mdl - | \
    sed "s/<TestMode> F/<TestMode> T/g" | sed "s/BatchNormComponent/BatchNormTestComponent/g" | \
    nnet3-copy --nnet-config=$dir/change.config - $dir/0.raw
fi


if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$train_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_train_dev_nodup_for_nas$suffix \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --temperature_schedule \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 3 \
    --trainer.optimization.num-jobs-initial 1 \
    --trainer.optimization.num-jobs-final 1 \
    --trainer.optimization.initial-effective-lrate 0.00025\
    --trainer.optimization.final-effective-lrate 0.000025 \
    --trainer.max-param-change 2.0 \
    --trainer.input-model $dir/0.raw \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set} \
    --tree-dir $treedir \
    --lat-dir exp/tri4_lats_dev_nodup_for_nas$suffix \
    --dir $dir  || exit 1;

fi

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_sw1_tg $dir $dir/graph_sw1_tg
fi


graph_dir=$dir/graph_sw1_tg
iter_opts=
if [ ! -z $decode_iter ]; then
  iter_opts=" --iter $decode_iter "
fi
if [ $stage -le 15 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in eval2000 rt03 rt02; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
          $graph_dir data/${decode_set}_fbk_40 \
          $dir/decode_${decode_set}_fbk_40${decode_iter:+_$decode_iter}_sw1_tg || exit 1;
     
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_sw1_{tg,fsh_fg} data/${decode_set}_fbk_40 \
        $dir/decode_${decode_set}_fbk_40${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;

      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi


# generate top model parameter size
nnet3-am-copy --binary=false $dir/final.mdl $dir/final_txt.mdl
python local/chain_NAS/scripts/bottleneckdim_search_top_model_size.py $dir top 'tdnn'
