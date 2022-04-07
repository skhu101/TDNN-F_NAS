// nnet3/nnet-convolutional-component.h

// Copyright      2017  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_NNET3_NNET_CONVOLUTIONAL_COMPONENT_H_
#define KALDI_NNET3_NNET_CONVOLUTIONAL_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/natural-gradient-online.h"
#include "nnet3/convolution.h"
#include <iostream>

namespace kaldi {
namespace nnet3 {

/// @file  nnet-convolutional-component.h
///
/// This file can be viewed as 'overflow' from nnet-general-component.h.
/// It contains a number of components which implement some kind of
/// convolution.


/**
   TdnnDARTSV3Component is a more memory-efficient alternative to manually splicing
   several frames of input and then using a NaturalGradientAffineComponent or
   a LinearComponent.  It does the splicing of the input itself, using
   mechanisms similar to what TimeHeightConvolutionComponent uses.  The
   implementation is in nnet-tdnn-component.cc

   Parameters inherited from UpdatableComponent (see comment above declaration of
   UpdadableComponent in nnet-component-itf.h for details):
       learning-rate, learning-rate-factor, max-change

   Important parameters:

     input-dim         The input feature dimension (before splicing).

     output-dim        The output feature dimension

     time-offsets     E.g. time-offsets=-1,0,1 or time-offsets=-3,0,3.
                      The time offsets that we require at the input to produce a given output.
                      comparable to the offsets used in TDNNs.  They
                      must be unique (no repeats).
     use-bias         Defaults to true, but set to false if you want this to
                      be linear rather than affine in its input.


  Extra parameters:
    orthonormal-constraint=0.0 If you set this to 1.0, then the linear_params_
                      matrix will be (approximately) constrained during training
                      to have orthonormal rows (or columns, whichever is
                      fewer).. it turns out the real name for this is a
                      "semi-orthogonal" matrix.  You can choose a positive
                      nonzero value different than 1.0 to have a scaled
                      semi-orthgonal matrix, i.e. with singular values at the
                      selected value (e.g. 0.5, or 2.0).  This is not enforced
                      inside the component itself; you have to call
                      ConstrainOrthonormal() from the training code to do this.
                      All this component does is return the
                      OrthonormalConstraint() value.  If you set this to a
                      negative value, it's like saying "for any value", i.e. it
                      will constrain the parameter matrix to be closer to "any
                      alpha" times a semi-orthogonal matrix, without changing
                      its overall norm.


   Initialization parameters:
      param-stddev    Standard deviation of the linear parameters of the
                      convolution.  Defaults to
                      sqrt(1.0 / (input-dim * the number of time-offsets))
      bias-stddev     Standard deviation of bias terms.  default=0.0.
                      You should not set this if you set use-bias=false.


   Natural-gradient related options are below; you won't normally have to
   set these as the defaults are reasonable.

      use-natural-gradient e.g. use-natural-gradient=false (defaults to true).
                       You can set this to false to disable the natural gradient
                       updates (you won't normally want to do this).
      rank-out         Rank used in low-rank-plus-unit estimate of the Fisher-matrix
                       factor that has the dimension (num-rows of linear_params_),
                       which equals output_dim.  It
                       defaults to the minimum of 80, or half of the output dim.
      rank-in          Rank used in low-rank-plus-unit estimate of the Fisher
                       matrix factor which has the dimension (num-cols of the
                       parameter matrix), which is input-dim times the number of
                       time offsets.  It defaults to the minimum of 20, or half the
                       num-rows of the parameter matrix.
      num-samples-history
                      This becomes the 'num_samples_history'
                      configuration value of the natural gradient objects.  The
                      default value is 2000.0.

 */
class TdnnDARTSV3Component: public UpdatableComponent {
 public:

  // The use of this constructor should only precede InitFromConfig()
  TdnnDARTSV3Component();

  // Copy constructor
  TdnnDARTSV3Component(const TdnnDARTSV3Component &other);

  virtual int32 InputDim() const {
    return linear_params_.NumCols() / static_cast<int32>(time_offsets_.size());
  }

  virtual int32 OutputDim() const { return linear_params_.NumRows(); }

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "TdnnDARTSV3Component"; }
  virtual int32 Properties() const {
    return kUpdatableComponent|kReordersIndexes|kBackpropAdds|
        (bias_params_.Dim() == 0 ? kPropagateAdds : 0)|
        kBackpropNeedsInput|kUsesMemo;
  }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void DeleteMemo(void *memo) const {
    delete static_cast<CuMatrix<BaseFloat>*>(memo);
  }

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Component* Copy() const {
    return new TdnnDARTSV3Component(*this);
  }


  // Some functions that are only to be reimplemented for GeneralComponents.

  // This ReorderIndexes function may insert 'blank' indexes (indexes with
  // t == kNoTime) as well as reordering the indexes.  This is allowed
  // behavior of ReorderIndexes functions.
  virtual void ReorderIndexes(std::vector<Index> *input_indexes,
                              std::vector<Index> *output_indexes) const;

  virtual void GetInputIndexes(const MiscComputationInfo &misc_info,
                               const Index &output_index,
                               std::vector<Index> *desired_indexes) const;

  // This function returns true if at least one of the input indexes used to
  // compute this output index is computable.
  virtual bool IsComputable(const MiscComputationInfo &misc_info,
                            const Index &output_index,
                            const IndexSet &input_index_set,
                            std::vector<Index> *used_inputs) const;

  virtual ComponentPrecomputedIndexes* PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const;

  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);
  virtual void FreezeNaturalGradient(bool freeze);


  class PrecomputedIndexes: public ComponentPrecomputedIndexes {
   public:
    PrecomputedIndexes() { }
    PrecomputedIndexes(const PrecomputedIndexes &other):
        row_stride(other.row_stride), row_offsets(other.row_offsets) { }
    virtual PrecomputedIndexes *Copy() const;
    virtual void Write(std::ostream &os, bool binary) const;
    virtual void Read(std::istream &os, bool binary);
    virtual std::string Type() const {
      return "TdnnDARTSV3ComponentPrecomputedIndexes";
    }
    virtual ~PrecomputedIndexes() { }


    // input_row_stride is the stride (in number of rows) we have to take in the
    // input matrix each time we form a sub-matrix that will be part of the
    // input to the tdnn operation.  Normally this will be 1, but it may be,
    // for example, 3 in layers where we do subsampling.
    int32 row_stride;

    // 'row_offsets' is of the same dimension as time_offsets_.  Each element
    // describes the row offset (in the input matrix) of a sub-matrix, and each.
    // We will append together these sub-matrices (row-wise) to be the input to
    // the affine or linear transform.
    std::vector<int32> row_offsets;
  };

  CuMatrixBase<BaseFloat> &LinearParams() { return linear_params_; }

  // This allows you to resize the vector in order to add a bias where
  // there previously was none-- obviously this should be done carefully.
  CuVector<BaseFloat> &BiasParams() { return bias_params_; }

  BaseFloat OrthonormalConstraint() const { return orthonormal_constraint_; }

  void ConsolidateMemory();

  void SetTempProportion(BaseFloat p) { temp_proportion_ = p; }

  // Call this with 'true' to set 'test mode' where the behavior is different
  // from normal mode.
  void SetTestMode(bool test_mode) { test_mode_ = test_mode; }

  //void AddBatchCount() { *batch_count += 1; }
 //protected:
  // This is true if we want a different behavior for inference from  that for
  // training.
  bool test_mode_;
  
 private:

  bool use_gumbel_;

  bool use_entropy_;

  bool free_select_;

  bool update_alpha_;

  bool update_theta_;

  bool uniform_sample_;

  BaseFloat temp_proportion_;

  // This static function is a utility function that extracts a CuSubMatrix
  // representing a subset of rows of 'input_matrix'.
  // The numpy syntax would be:
  //   return input_matrix[row_offset:row_stride:num_output_rows*row_stride,:]
  static CuSubMatrix<BaseFloat> GetInputPart(
      const CuMatrixBase<BaseFloat> &input_matrix,
      int32 num_output_rows,
      int32 row_stride,
      int32 row_offset);

  // see the definition for more explanation.
  static void ModifyComputationIo(time_height_convolution::ConvolutionComputationIo *io);

  void Check() const;

  // Function that updates linear_params_, and bias_params_ if present, which
  // uses the natural gradient code.
  void UpdateNaturalGradient(
      const PrecomputedIndexes &indexes,
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv,
      const CuMatrixBase<BaseFloat> &linear_params_temp_,
      const CuVector<BaseFloat> &bias_params_temp_,
      const CuVector<BaseFloat> &coef_temp_,
      const int32 share_offset_index_temp_,
      const bool use_gumbel_temp_,
      const bool uniform_sample_temp_,
      const bool use_entropy_temp_,
      const bool free_select_temp_,
      const bool update_alpha_temp_,
      const bool update_theta_temp_, 
      const BaseFloat temp_proportion_temp_);

  // Function that updates linear_params_, and bias_params_ if present, which
  // does not use the natural gradient code.
  void UpdateSimple(
      const PrecomputedIndexes &indexes,
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);


  // time_offsets_ is the list of time-offsets of the input that
  // we append together; it will typically be (-1,0,1) or (-3,0,3).
  std::vector<int32> time_offsets_;

  // the linear parameters of the network; its NumRows() is the output
  // dim, and its NumCols() equals the input dim times time_offsets_.size().
  CuMatrix<BaseFloat> linear_params_;

  // the bias parameters if this is an affine transform, or the empty vector if
  // this is a linear operation (i.e. use-bias == false in the config).
  CuVector<BaseFloat> bias_params_;

  // If nonzero, this controls how we apply an orthonormal constraint to the
  // parameter matrix; see docs for ConstrainOrthonormal() in nnet-utils.h.
  // This class just returns the value via the OrthonormalConstraint() function;
  // it doesn't actually do anything with it directly.
  BaseFloat orthonormal_constraint_;

  // Controls whether or not the natural-gradient is used.  Note: even if this
  // is true, if is_gradient_ (from the UpdatableComponent base class) is true,
  // we'll do the 'simple' update that doesn't include natural gradient.
  bool use_natural_gradient_;

  // Preconditioner for the input space, of dimension linear_params_.NumCols() +
  // 1 (the 1 is for the bias).  As with other natural-gradient objects, it's
  // not stored with the model on disk but is reinitialized each time we start
  // up.
  OnlineNaturalGradient preconditioner_in_;

  // Preconditioner for the output space, of dimension
  // linear_params_.NumRows().
  OnlineNaturalGradient preconditioner_out_;
};


} // namespace nnet3
} // namespace kaldi


#endif
