. ./cmd.sh
. ./path.sh

#mfcc
utils/subset_data_dir.sh --first data/train_nodup 10000 data/train_dev_nodup_for_nas 
n=$[`cat data/train_nodup/segments | wc -l` - 10000]
utils/subset_data_dir.sh --last data/train_nodup $n data/train_nodev_nodup_for_nas


for train_set in train_dev_nodup_for_nas train_nodev_nodup_for_nas ; do
    echo "$0: preparing directory for speed-perturbed data"
    utils/data/perturb_data_dir_speed_3way.sh --always-include-prefix true \
           data/${train_set} data/${train_set}_sp

    echo "$0: creating MFCC features for low-resolution speed-perturbed data"
    mfccdir=mfcc_perturbed
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 50 \
                       data/${train_set}_sp exp/make_mfcc/${train_set}_sp $mfccdir
    steps/compute_cmvn_stats.sh data/${train_set}_sp exp/make_mfcc/${train_set}_sp $mfccdir
    utils/fix_data_dir.sh data/${train_set}_sp
done

for dataset in  train_dev_nodup_for_nas_sp train_nodev_nodup_for_nas_sp ; do
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires

    utils/data/perturb_data_dir_volume.sh data/${dataset}_hires

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/${dataset} $mfccdir;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh data/${dataset}_hires;
done

for train_set in train_dev_nodup_for_nas_sp train_nodev_nodup_for_nas_sp ; do
    utils/data/modify_speaker_info.sh --utts-per-spk-max 2 data/${train_set}_hires data/${train_set}_max2_hires
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
        data/${train_set}_max2_hires exp/nnet3/extractor exp/nnet3/ivectors_$train_set || exit 1;
done


#fbank
utils/subset_data_dir.sh --first data/train_fbk_40_nodup 10000 data/train_fbk_40_dev_nodup_for_nas # 5hr 6min
n=$[`cat data/train_fbk_40_nodup/segments | wc -l` - 10000]
utils/subset_data_dir.sh --last data/train_fbk_40_nodup $n data/train_fbk_40_nodev_nodup_for_nas

for train_set in train_fbk_40_dev_nodup_for_nas train_fbk_40_nodev_nodup_for_nas ; do
    echo "$0: preparing directory for speed-perturbed data"
    utils/data/perturb_data_dir_speed_3way.sh --always-include-prefix true \
           data/${train_set} data/${train_set}_sp

    echo "$0: creating MFCC features for low-resolution speed-perturbed data"
    mfccdir=fbk_40_perturbed
    steps/make_fbank_40.sh --cmd "$train_cmd" --nj 50 \
                       data/${train_set}_sp exp/make_fbk_40/${train_set}_sp $mfccdir
    steps/compute_cmvn_stats.sh data/${train_set}_sp exp/make_fbk_40/${train_set}_sp $mfccdir
    utils/fix_data_dir.sh data/${train_set}_sp
done



# generate lattice 

steps/align_fmllr.sh --nj 50 --cmd "$train_cmd" data/train_nodev_nodup_for_nas_sp data/lang exp/tri4 exp/tri4_ali_nodev_nodup_for_nas_sp
steps/align_fmllr.sh --nj 50 --cmd "$train_cmd" data/train_dev_nodup_for_nas_sp data/lang exp/tri4 exp/tri4_ali_dev_nodup_for_nas_sp

nj=$(cat exp/tri4_ali_nodev_nodup_for_nas_sp/num_jobs) || exit 1;
steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/train_nodev_nodup_for_nas_sp data/lang exp/tri4 exp/tri4_lats_nodev_nodup_for_nas_sp
rm exp/tri4_lats_nodev_nodup_for_nas_sp/fsts.*.gz # save space

nj=$(cat exp/tri4_ali_dev_nodup_for_nas_sp/num_jobs) || exit 1;
steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/train_dev_nodup_for_nas_sp data/lang exp/tri4 exp/tri4_lats_dev_nodup_for_nas_sp
rm exp/tri4_lats_dev_nodup_for_nas_sp/fsts.*.gz # save space

