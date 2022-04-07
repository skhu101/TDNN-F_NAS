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
parent_path=$1
flops_coef=$2
child_type=$3
top_id=$4
#offset=$4
gpu_id=$5
offset_0=$6
offset_1=$7
offset_2=$8
offset_3=$9
offset_4=${10}
offset_5=${11}
offset_6=${12}
offset_7=${13}
offset_8=${14}
offset_9=${15}
offset_10=${16}
offset_11=${17}
offset_12=${18}
offset_13=${19}
offset_14=${20}
offset_15=${21}
offset_16=${22}
offset_17=${23}
offset_18=${24}
offset_19=${25}
offset_20=${26}
offset_21=${27}
offset_22=${28}
offset_23=${29}
offset_24=${30}
offset_25=${31}
offset_26=${32}
offset_27=${33}
offset_type=${34}
egs_dir=${35}

# configs for 'chain'
stage=0
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
#common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

test_online_decoding=false  # if true, it will run the last decoding stage.

# End configuration section.
echo "$0 $@"  # Print the command line for logging

export CUDA_VISIBLE_DEVICES=${gpu_id}

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

suffix=
$speed_perturb && suffix=_sp
if [[ $parent_path =~ "95onehotpretrain_cvupdate_gumbel" ]]; then
    dir=exp/chain_NAS/tdnn_DARTS_${offset_type}_bottleneckCBshare_95onehotpretrain_cvupdate_gumbel_flops_${flops_coef}_Child_Top${top_id}${affix}${suffix}
elif [[ $parent_path =~ "95onehotpretrain_cvupdate_softmax" ]]; then
    dir=exp/chain_NAS/tdnn_DARTS_${offset_type}_bottleneckCBshare_95onehotpretrain_cvupdate_softmax_flops_${flops_coef}_Child_Top${top_id}${affix}${suffix}
elif [[ $parent_path =~ "gumbel" ]]; then
    dir=exp/chain_NAS/tdnn_DARTS_${offset_type}_bottleneckCBshare_gumbel_Child_Top${top_id}${affix}${suffix}
elif [[ $parent_path =~ "softmax" ]]; then
    dir=exp/chain_NAS/tdnn_DARTS_${offset_type}_bottleneckCBshare_softmax_Child_Top${top_id}${affix}${suffix}
fi

common_egs_dir=$egs_dir

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

train_set=train_fbk_40_nodup$suffix
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
  # enlarge 4 times
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  #fixed-affine-layer name=lda input=Append(-1,0,1) affine-transform-file=$dir/configs/lda.mat
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $affine_opts dim=1536
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  linear-component name=prefinal-l dim=256 $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
  mv $dir/configs/final.config $dir/configs/final.config_temp
  mv $dir/configs/ref.config $dir/configs/ref.config_temp
  ~/anaconda3/bin/python local/chain_NAS/scripts/generate_optimal_stride.py $dir/configs $offset_0 $offset_1 $offset_2 $offset_3 $offset_4 $offset_5 $offset_6 $offset_7 $offset_8 $offset_9 $offset_10 $offset_11 $offset_12 $offset_13 $offset_14 $offset_15 $offset_16 $offset_17 $offset_18 $offset_19 $offset_20 $offset_21 $offset_22 $offset_23 $offset_24 $offset_25 $offset_26 $offset_27

  mv $dir/configs/final.config $dir/configs/final.config_temp
  mv $dir/configs/ref.config $dir/configs/ref.config_temp
  #nnet3-am-copy --binary=false $parent_path/final.mdl $parent_path/final_txt.mdl
  ~/anaconda3/bin/python local/chain_NAS/scripts/generate_top_list_bottleneckdim.py $parent_path $child_type $top_id $dir/configs/ 8 tdnn
  steps/nnet3/xconfig_to_configs_cal_info.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$train_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_train_nodup$suffix \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
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
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set} \
    --tree-dir $treedir \
    --lat-dir exp/tri4_lats_nodup$suffix \
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
rm -rf $dir/decode_eval2000_fbk_40_sw1_tg $dir/decode_rt02_fbk_40_sw1_tg $dir/decode_rt03_fbk_40_sw1_tg
