# TDNN-F_NAS
This repository contains the Kaldi LF-MMI implementation of the paper **Neural Architecture Search for LF-MMI Trained Time Delay Neural Networks**, IEEE/ACM Transactions on Audio Speech and Language (TASLP).

By Shoukang Hu, Xurong Xie, Mingyu Cui*, Jiajun Deng*, Shansong Liu, Jianwei Yu, Mengzhe Geng, Xunying Liu, Helen Meng

[Paper](https://arxiv.org/abs/2201.03943)

## Getting Started
* Install [Kaldi](https://github.com/kaldi-asr/kaldi)
* Clone the repo:
  ```
  git clone https://github.com/skhu101/TDNN-F_NAS.git
  ```
  
### Usage
Step 1: 

* Copy the TdnnDARTSV3Component in nnet-convolutional-component.h to kaldi/src/nnet3/nnet-convolutional-component.h 
* Copy the TdnnDARTSV3Component in nnet-tdnn-component.cc to kaldi/src/nnet3/nnet-tdnn-component.cc 

* Copy the OnehotFunctionComponent in nnet-simple-component.h to kaldi/src/nnet3/nnet-simple-component.h 
* Copy the OnehotFunctionComponent in nnet-simple-component.cc to kaldi/src/nnet3/nnet-simple-component.cc

* Copy the CopyNComponent in nnet-simple-component.h to kaldi/src/nnet3/nnet-simple-component.h 
* Copy the CopyNComponent in nnet-simple-component.cc to kaldi/src/nnet3/nnet-simple-component.cc

* Copy the GumbelSoftmaxFlopsComponent in nnet-simple-component.h to kaldi/src/nnet3/nnet-simple-component.h 
* Copy the GumbelSoftmaxFlopsComponent in nnet-simple-component.cc to kaldi/src/nnet3/nnet-simple-component.cc

* Copy the SoftmaxFlopsComponent in nnet-simple-component.h to kaldi/src/nnet3/nnet-simple-component.h 
* Copy the SoftmaxFlopsComponent in nnet-simple-component.cc to kaldi/src/nnet3/nnet-simple-component.cc


* Copy the following lines to the corresponding location in kaldi/src/nnet3/nnet-component-itf.cc
```shell
  } else if (cpi_type == "TdnnDARTSV3ComponentPrecomputedIndexes") {
    ans = new TdnnDARTSV3Component::PrecomputedIndexes(); 

  } else if (component_type == "TdnnDARTSV3Component") {
    ans = new TdnnDARTSV3Component(); 
    
  } else if (component_type == "OnehotFunctionComponent") {
    ans = new OnehotFunctionComponent();   
    
  } else if (component_type == "CopyNComponent") {
    ans = new CopyNComponent();
    
  } else if (component_type == "SoftmaxFlopsComponent") {
    ans = new SoftmaxFlopsComponent();

  } else if (component_type == "GumbelSoftmaxFlopsComponent") {
    ans = new GumbelSoftmaxFlopsComponent();

```

* Copy the kaldi/src/nnet3/nnet-utils.cc to kaldi/src/nnet3/nnet-utils.cc 

* If you want to add the specific code, you can use the following command:
```shell
cd src/nnet3/
grep -r "TdnnDARTSV3Component" .
grep -r "OnehotFunctionComponent" .
grep -r "CopyNComponent" .
grep -r "SoftmaxFlopsComponent" .
grep -r "GumbelSoftmaxFlopsComponent" .
```

* complie the new source file 
```shell
cd kaldi/src/nnet3/
make -j 20
```

Step 2: 

* copy the files in steps to kaldi/egs/swbd/s5c/steps; copy the files in local/chain_NAS to kaldi/egs/swbd/s5c/local/chain_NAS

Step 3: 
* run the factored TDNN model using the following command
```shell
cd kaldi/egs/swbd/s5c
bash local/chain/tuning/run_tdnn_7q.sh
```

Step 4:
* split the training data into a ration of 95:5 by using the command in src/nnet3/Prepare_NAS_data.sh
```shell
bash Prepare_NAS_data.sh
```

Step 5: 

#### manual system
```shell
bash local/chain_NAS/run_tdnn_7q_fbk_40_manual.sh --offset 6 --bottleneckdim 160
```

Step 6: 

#### context offset pipeline search
* 95% pretrain
```shell
bash local/chain_NAS/run_TDNN_DARTSV3_fbk_stride_pretrain.sh 7
```

* 5% cv update
```shell
bash local/chain_NAS/run_TDNN_DARTSV3_fbk_stride_cvupdate.sh offset-len parent-path use-gumbel
```

>For example:
  
```
  # gumbel 5% cv update
  bash local/chain_NAS/run_TDNN_DARTSV3_fbk_stride_cvupdate.sh --offset-len 7 --parent-path exp/chain_NAS/tdnn_DARTSV3_context_offset7_95peronehotpretrain_fbk_40_iv_7q_sp --use-gumbel true
  # softmax 5% cv update
  bash local/chain_NAS/run_TDNN_DARTSV3_fbk_stride_cvupdate.sh --offset-len 7 --parent-path exp/chain_NAS/tdnn_DARTSV3_context_offset7_95peronehotpretrain_fbk_40_iv_7q_sp --use-gumbel false
```

* train the top1 model in the context offset (6) search
```shell
  bash local/chain_NAS/run_TDNN_DARTS_Child_mod_fbk.sh parent_path top top_id offset-len gpu_id
```

>For example:
```shell
  bash local/chain_NAS/run_TDNN_DARTS_Child_mod_fbk.sh exp/chain_NAS/tdnn_DARTSV3_offset7_fbk_40_iv_7q_sp_95onehotpretrain_cvupdate_gumbel top 1 7 1
```

Step 7:

#### bottleneck dimension pipeline search with context offset 4
* 95% pretrain
```shell
bash local/chain_NAS/run_TDNNf_DARTS_mod_fbk_bottleneckCBshare_95onehottrain.sh offset-type gpu-id
```

>For example:
```shell
bash local/chain_NAS/run_TDNNf_DARTS_mod_fbk_bottleneckCBshare_95onehottrain.sh 4 0 
```

* 5% cv update
```shell
bash local/chain_NAS/run_TDNNf_DARTS_mod_fbk_bottleneckCBshare_cvupdate_flopsconstraint.sh parent-path use-gumbel flops-coef
```

>For example:

```
  # pipelinegumbel 5% cv update
  bash local/chain_NAS/run_TDNNf_DARTS_mod_fbk_bottleneckCBshare_cvupdate_flopsconstraint.sh --parent-path exp/chain_NAS/tdnn_DARTS_bottleneckCBshare_95onehottrain_25_50_80_100_120_160_200_240_fbk_40_iv_7q_sp --use-gumbel true --flops-coef 1e-3
  # pipelinesoftmax 5% cv update
  bash local/chain_NAS/run_TDNNf_DARTS_mod_fbk_bottleneckCBshare_cvupdate_flopsconstraint.sh --parent-path exp/chain_NAS/tdnn_DARTS_bottleneckCBshare_95onehottrain_25_50_80_100_120_160_200_240_fbk_40_iv_7q_sp --use-gumbel false --flops-coef 0
```

<!-- * gumbel/softmax update 
```shell
bash local/chain_NAS/run_TDNNf_DARTS_mod_fbk_bottleneckCBshare.sh offset-len use-gumbel gpu-id
```

>For example:
```shell
  bash local/chain_NAS/run_TDNNf_DARTS_mod_fbk_bottleneckCBshare.sh 3 true 1
``` -->

* train the top1 model in bottleneck dim search (25,50,80,100,120,160,200,240)
```shell
bash local/chain_NAS/run_TDNN_DARTS_bottleneckdim_Child_mod_fbk.sh parent_path fops_coef child_type top_id gpu_id
```

>For example:
```shell
  bash local/chain_NAS/run_TDNN_DARTS_bottleneckdim_Child_mod_fbk.sh exp/chain_NAS/tdnn_DARTS_bottleneckCBshare_95onehotpretrain_cvupdate_gumbel_flopsconstraint_1e-3 1e-3 top 1 1
```

* calculate top model parameter size 
```shell
nnet3-am-copy --binary=false model_dir/final.mdl model_dir/final_txt.mdl
python local/chain_NAS/scripts/bottleneckdim_search_top_model_size.py model_dir top network-type
```

>For example:
```shell
  nnet3-am-copy --binary=false exp/chain_NAS/tdnn_DARTS_bottleneckCBshare_95onehotpretrain_cvupdate_gumbel_flopsconstraint_1e-3/final.mdl exp/chain_NAS/tdnn_DARTS_bottleneckCBshare_95onehotpretrain_cvupdate_gumbel_flopsconstraint_1e-3/final_txt.mdl
  python local/chain_NAS/scripts/bottleneckdim_search_top_model_size.py exp/chain_NAS/tdnn_DARTS_bottleneckCBshare_95onehotpretrain_cvupdate_gumbel_flopsconstraint_1e-3 top 'tdnn'
```

Step 8:
#### bottleneck dimension pipeline search based on the optimal context offset learned in step 6

* 95% pretrain
```shell
bash local/chain_NAS/run_TDNNf_DARTS_mod_fbk_optimal_context_offset_bottleneckCBshare_95onehottrain.sh offset-type gpu-id offset0 offset1 ... offset13
```

>For example:


```shell
  # pipeline gumbel 95% pretrain
  bash local/chain_NAS/run_TDNNf_DARTS_mod_fbk_optimal_context_offset_bottleneckCBshare_95onehottrain.sh pipegumbel_context_offset6_top1 1 -2 2 -2 4 -5 5 -6 6 -6 5 -6 6 -6 6 -6 6 -6 6 -6 6 -6 6 -6 6 -6 6 -6 6
  # piepline softmax 95% pretrain
  bash local/chain_NAS/run_TDNNf_DARTS_mod_fbk_optimal_context_offset_bottleneckCBshare_95onehottrain.sh pipesoftmax_context_offset6_top1 0 -1 2 -2 2 -2 5 -3 6 -4 5 -5 6 -6 6 -6 6 -6 5 -6 6 -6 6 -6 6 -6 6 -6 6
```

* 5% cv update
>For example:


```shell
  # pipeline gumbel 5% cv update
  bash local/chain_NAS/run_TDNNf_DARTS_mod_fbk_optimal_context_offset_bottleneckCBshare_cvupdate_flopsconstraint.sh --offset-type pipegumbel_context_offset6_top1 --parent-path exp/chain_NAS/tdnn_DARTS_pipegumbel_context_offset6_top1_bottleneckCBshare_95onehottrain_25_50_80_100_120_160_200_240_fbk_40_iv_7q_sp --use-gumbel true --flops-coef 1e-1
  # piepline softmax 5% cv update
  bash local/chain_NAS/run_TDNNf_DARTS_mod_fbk_optimal_context_offset_bottleneckCBshare_cvupdate_flopsconstraint.sh --offset-type pipesoftmax_context_offset6_top1 --parent-path exp/chain_NAS/tdnn_DARTS_pipesoftmax_context_offset6_top1_bottleneckCBshare_95onehottrain_25_50_80_100_120_160_200_240_fbk_40_iv_7q_sp --use-gumbel false --flops-coef 1e-1
```

* train the top1 model in bottleneck dim search (25,50,80,100,120,160,200,240) based on the optimal context offset

searched model from pipeline gumbel 5% cv update
```shell
bash local/chain_NAS/run_TDNN_DARTS_optimal_context_offset_bottleneckdim_Child_mod_fbk.sh parent_path fops_coef child_type top_id gpu_id offset0  ... offset27 offset_type egs_dir
```

>For exmaple:
```shell
  bash local/chain_NAS/run_TDNN_DARTS_optimal_context_offset_bottleneckdim_Child_mod_fbk.sh exp/chain_NAS/tdnn_DARTS_pipegumbel_context_offset6_top1_bottleneckCBshare_95onehotpretrain_cvupdate_gumbel_flopsconstraint_3e-1 3e-1 top 1 0 -2 2 -2 4 -5 5 -6 6 -6 5 -6 6 -6 6 -6 6 -6 6 -6 6 -6 6 -6 6 -6 6 -6 6 pipegumbel_context_offset6_top1 /project_bdda4/bdda/skhu/toolkits/kaldi/egs/swbd_wav/s5c/exp/chain_NAS/tdnn_DARTS_context_offset7_95onehotpretrain_cvupdate_gumbel_Child_Top1_fbk_40_iv_7q_sp/egs

  bash local/chain_NAS/run_TDNN_DARTS_optimal_context_offset_bottleneckdim_Child_mod_fbk.sh exp/chain_NAS/tdnn_DARTS_pipesoftmax_context_offset6_top1_bottleneckCBshare_95onehotpretrain_cvupdate_softmax_flopsconstraint_1e-1 1e-1 top 1 0 -1 2 -2 2 -2 5 -3 6 -4 5 -5 6 -6 6 -6 6 -6 5 -6 6 -6 6 -6 6 -6 6 -6 6 pipesoftmax_context_offset6_top1 exp/chain_NAS/tdnn_DARTS_context_offset7_95onehotpretrain_cvupdate_softmax_Child_Top1_fbk_40_iv_7q_sp/egs
```

### Citation
If you find our codes or trained models useful in your research, please consider to star our repo and cite our paper:

    @article{hu2022neural,
      title={Neural architecture search for LF-MMI trained time delay neural networks},
      author={Hu, Shoukang and Xie, Xurong and Cui, Mingyu and Deng, Jiajun and Liu, Shansong and Yu, Jianwei and Geng, Mengzhe and Liu, Xunying and Meng, Helen M},
      journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
      year={2022},
      publisher={IEEE}
    }

<!--     @inproceedings{hu2021neural,
      title={Neural Architecture Search for LF-MMI Trained Time Delay Neural Networks},
      author={Hu, Shoukang and Xie, Xurong and Liu, Shansong and Cui, Mingyu and Geng, Mengzhe and Liu, Xunying and Meng, Helen},
      booktitle={International Conference on Acoustics, Speech, and Signal Processing},
      pages={6758--6762},
      year={2021}
    } -->
