/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "feature_provider.h"

#include "audio_provider.h"
#include "micro_features_micro_features_generator.h"
#include "micro_features_micro_model_settings.h"

float* FillFeatureBuffer(float* input_feature_buffer){
  input_feature_buffer = {9.80580370e-01 9.95648267e-02 9.12149037e-01 3.16358783e-03
 2.01203996e-02 2.01203996e-02 4.45788585e-04 3.15653886e-01
 3.10521403e-02 9.12149037e-01 9.87149652e-01 8.36772578e-02
 8.62832051e-01 1.28903323e-02 5.76937706e-03 5.76937706e-03
 1.05818115e-04 5.08647758e-01 6.10697111e-03 8.62832051e-01
 9.90974335e-01 2.21812730e-02 6.60626352e-01 2.44961325e-03
 1.95529650e-02 1.95529650e-02 4.23623950e-04 4.15288125e-01
 2.53718840e-02 6.60626352e-01 1.16735378e-01 4.83901570e-04
 4.39536966e-04 2.48039869e-02 2.73325914e-04 2.73325914e-04
 1.56500001e-07 4.76488983e-01 1.66694970e-04 4.39536966e-04
 7.88196755e-32 1.49018576e-33 3.01862816e-33 2.47231194e-02
 2.46610879e-34 2.46610879e-34 1.21388465e-67 3.95472611e-01
 1.73300565e-34 3.01862816e-33 9.03147850e-01 3.28390942e-02
 1.46899796e-01 5.75652798e-02 3.70167737e-02 3.70167737e-02
 1.46099378e-03 5.08397289e-01 2.76867262e-02 1.46899796e-01
 1.84001110e-01 9.17207741e-02 4.00768558e-01 5.05028807e-02
 2.04835446e-01 2.04835446e-01 4.19741884e-02 1.76589094e-01
 1.99773002e-01 4.00768558e-01 2.34100205e-03 9.98957731e-04
 6.19903365e-04 8.92526686e-02 7.21029136e-04 7.21029136e-04
 5.25391825e-07 4.30017306e-01 3.78371701e-04 6.19903365e-04
 4.31120127e-07 1.73401022e-08 2.47726258e-08 5.76758973e-02
 5.55625191e-09 5.55625191e-09 2.91185194e-17 8.72918045e-01
 3.22401834e-09 2.47726258e-08 9.81176971e-01 2.50765264e-02
 7.52940060e-01 4.72668282e-02 1.53380921e-02 1.53380921e-02
 2.62794352e-04 3.02159123e-01 1.92956155e-02 7.52940060e-01
 1.95290008e-02 4.21546857e-01 3.44167868e-01 4.03191949e-03
 5.31554012e-01 5.31554012e-01 2.82550313e-01 3.32329851e-01
 5.45095615e-01 3.44167868e-01}
 
 return input_feature_buffer;
}
// constructor initializes data array to zeros
FeatureProvider::FeatureProvider(int feature_size, int8_t* feature_data)
    : feature_size_(feature_size),
      feature_data_(feature_data),
      is_first_run_(true) {
  // Initialize the feature data to default values.
  for (int n = 0; n < feature_size_; ++n) {
    feature_data_[n] = 0;
  }
}

FeatureProvider::~FeatureProvider() {}

TfLiteStatus FeatureProvider::PopulateFeatureData(
    tflite::ErrorReporter* error_reporter, int32_t last_time_in_ms,
    int32_t time_in_ms, int* how_many_new_slices) {
  if (feature_size_ != kFeatureElementCount) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Requested feature_data_ size %d doesn't match %d",
                         feature_size_, kFeatureElementCount);
    return kTfLiteError;
  }

  // Quantize the time into steps as long as each window stride, so we can
  // figure out which audio data we need to fetch.
  const int last_step = (last_time_in_ms / kFeatureSliceStrideMs);
  const int current_step = (time_in_ms / kFeatureSliceStrideMs);

  int slices_needed = current_step - last_step;
  // If this is the first call, make sure we don't use any cached information.
  if (is_first_run_) {
    TfLiteStatus init_status = InitializeMicroFeatures(error_reporter);
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    is_first_run_ = false;
    slices_needed = kFeatureSliceCount;
  }
  if (slices_needed > kFeatureSliceCount) {
    slices_needed = kFeatureSliceCount;
  }
  *how_many_new_slices = slices_needed;

  const int slices_to_keep = kFeatureSliceCount - slices_needed;
  const int slices_to_drop = kFeatureSliceCount - slices_to_keep;
  // If we can avoid recalculating some slices, just move the existing data
  // up in the spectrogram, to perform something like this:
  // last time = 80ms          current time = 120ms
  // +-----------+             +-----------+
  // | data@20ms |         --> | data@60ms |
  // +-----------+       --    +-----------+
  // | data@40ms |     --  --> | data@80ms |
  // +-----------+   --  --    +-----------+
  // | data@60ms | --  --      |  <empty>  |
  // +-----------+   --        +-----------+
  // | data@80ms | --          |  <empty>  |
  // +-----------+             +-----------+
  if (slices_to_keep > 0) {
    for (int dest_slice = 0; dest_slice < slices_to_keep; ++dest_slice) {
      int8_t* dest_slice_data =
          feature_data_ + (dest_slice * kFeatureSliceSize);
      const int src_slice = dest_slice + slices_to_drop;
      const int8_t* src_slice_data =
          feature_data_ + (src_slice * kFeatureSliceSize);
      for (int i = 0; i < kFeatureSliceSize; ++i) {
        dest_slice_data[i] = src_slice_data[i];
      }
    }
  }
  // Any slices that need to be filled in with feature data have their
  // appropriate audio data pulled, and features calculated for that slice.
  if (slices_needed > 0) {
    for (int new_slice = slices_to_keep; new_slice < kFeatureSliceCount;
         ++new_slice) {
      const int new_step = (current_step - kFeatureSliceCount + 1) + new_slice;
      const int32_t slice_start_ms = (new_step * kFeatureSliceStrideMs);
      int16_t* audio_samples = nullptr;
      int audio_samples_size = 0;
      // TODO(petewarden): Fix bug that leads to non-zero slice_start_ms
      GetAudioSamples(error_reporter, (slice_start_ms > 0 ? slice_start_ms : 0),
                      kFeatureSliceDurationMs, &audio_samples_size,
                      &audio_samples);
      if (audio_samples_size < kMaxAudioSampleSize) {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "Audio data size %d too small, want %d",
                             audio_samples_size, kMaxAudioSampleSize);
        return kTfLiteError;
      }
      int8_t* new_slice_data = feature_data_ + (new_slice * kFeatureSliceSize);
      size_t num_samples_read;
      TfLiteStatus generate_status = GenerateMicroFeatures(
          error_reporter, audio_samples, audio_samples_size, kFeatureSliceSize,
          new_slice_data, &num_samples_read);
      if (generate_status != kTfLiteOk) {
        return generate_status;
      }
    }
  }
  return kTfLiteOk;
}
