/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <TensorFlowLite.h>

#include "main_functions.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
//#include "constants.h"
#include "model.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
//int inference_count = 0;

constexpr int kTensorArenaSize = 200 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);

  if ((input->dims->size != 1) || (input->dims->data[0] != 11) ||
  (input->dims->data[0] != 10) || (input->type != kTfLiteFloat32)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }
  // Make sure the input has the properties we expect
//  micro_test::TF_LITE_MICRO_EXPECT_NE(nullptr, input);
//  micro_test::TF_LITE_MICRO_EXPECT_EQ(1, input->dims->size);
//  micro_test::TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
//  micro_test::TF_LITE_MICRO_EXPECT_EQ(110, input->dims->data[1]);
//  micro_test::TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);
  
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  // inference_count = 0;
}

// The name of this function is important for Arduino compatibility.
void loop() {
 float feature_buffer[11][10] = {
   {0.9890421, 0.111468054, 0.9532741, 0.0488911, 0.0073072645, 0.0073072645, 6.899725e-05, 0.33836108, 0.007248835, 0.9532741}, 
   {0.9771623, 0.09478671, 0.8691787, 0.034773305, 0.014809458, 0.014809458, 0.0004038004, 0.6272644, 0.012837556, 0.8691787}, 
   {0.95854825, 0.0138841905, 0.15692894, 0.023273712, 0.014867932, 0.014867932, 0.00025264578, 0.35475492, 0.014191338, 0.15692894}, 
   {0.1159372, 0.0005301315, 0.0004292443, 0.010490402, 0.00045734065, 0.00045734065, 3.4460643e-07, 0.45727947, 0.00035582457, 0.0004292443}, 
   {7.473063e-32, 4.055662e-33, 2.7790986e-33, 0.042500712, 1.0050182e-33, 1.0050182e-33, 0.0, 0.42498454, 6.414021e-34, 2.7790986e-33}, 
   {0.8735593, 0.08603359, 0.10240642, 0.027771521, 0.13362896, 0.13362896, 0.018151218, 0.48276207, 0.10830316, 0.10240642}, 
   {0.62360907, 0.18915448, 0.621753, 0.027151836, 0.25723293, 0.25723293, 0.066117615, 0.36665785, 0.34866726, 0.621753}, 
   {0.0020959722, 0.0009574314, 0.001291765, 0.034545664, 0.0009985382, 0.0009985382, 1.0047055e-06, 0.17116626, 0.0007413034, 0.001291765},
   {4.5866375e-07, 2.815845e-08, 4.895011e-08, 0.01965016, 7.141418e-09, 7.141418e-09, 4.8103243e-17, 0.68389165, 5.436633e-09, 4.895011e-08}, 
   {0.98903406, 0.030304093, 0.88752085, 0.03398187, 0.017289415, 0.017289415, 0.0003299018, 0.4221791, 0.024418829, 0.88752085}, 
   {0.24432209, 0.62170565, 0.39452726, 0.03759679, 0.41413438, 0.41413438, 0.1715089, 0.4274359, 0.3635621, 0.39452726}
   };

//  float feature_buffer[110] = {
//   9.80580370e-01, 9.95648267e-02, 9.12149037e-01, 3.16358783e-03, 
//   2.01203996e-02, 2.01203996e-02, 4.45788585e-04, 3.15653886e-01,
//   3.10521403e-02, 9.12149037e-01, 9.87149652e-01, 8.36772578e-02,
//   8.62832051e-01, 1.28903323e-02, 5.76937706e-03, 5.76937706e-03,
//   1.05818115e-04, 5.08647758e-01, 6.10697111e-03, 8.62832051e-01,
//   9.90974335e-01, 2.21812730e-02, 6.60626352e-01, 2.44961325e-03,
//   1.95529650e-02, 1.95529650e-02, 4.23623950e-04, 4.15288125e-01,
//   2.53718840e-02, 6.60626352e-01, 1.16735378e-01, 4.83901570e-04,
//   4.39536966e-04, 2.48039869e-02, 2.73325914e-04, 2.73325914e-04,
//   1.56500001e-07, 4.76488983e-01, 1.66694970e-04, 4.39536966e-04,
//   7.88196755e-32, 1.49018576e-33, 3.01862816e-33, 2.47231194e-02,
//   2.46610879e-34, 2.46610879e-34, 1.21388465e-67, 3.95472611e-01,
//   1.73300565e-34, 3.01862816e-33, 9.03147850e-01, 3.28390942e-02,
//   1.46899796e-01, 5.75652798e-02, 3.70167737e-02, 3.70167737e-02,
//   1.46099378e-03, 5.08397289e-01, 2.76867262e-02, 1.46899796e-01,
//   1.84001110e-01, 9.17207741e-02, 4.00768558e-01, 5.05028807e-02,
//   2.04835446e-01, 2.04835446e-01, 4.19741884e-02, 1.76589094e-01,
//   1.99773002e-01, 4.00768558e-01, 2.34100205e-03, 9.98957731e-04,
//   6.19903365e-04, 8.92526686e-02, 7.21029136e-04, 7.21029136e-04,
//   5.25391825e-07, 4.30017306e-01, 3.78371701e-04, 6.19903365e-04,
//   4.31120127e-07, 1.73401022e-08, 2.47726258e-08, 5.76758973e-02,
//   5.55625191e-09, 5.55625191e-09, 2.91185194e-17, 8.72918045e-01,
//   3.22401834e-09, 2.47726258e-08, 9.81176971e-01, 2.50765264e-02,
//   7.52940060e-01, 4.72668282e-02, 1.53380921e-02, 1.53380921e-02,
//   2.62794352e-04, 3.02159123e-01, 1.92956155e-02, 7.52940060e-01,
//   1.95290008e-02, 4.21546857e-01, 3.44167868e-01, 4.03191949e-03,
//   5.31554012e-01, 5.31554012e-01, 2.82550313e-01, 3.32329851e-01,
//   5.45095615e-01, 3.44167868e-01};

  for (int i=0; i<11; i++)
    for (int j=0; j<10; j++)
      input->data.f[i] = feature_buffer[i][j];

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x: %f\n",
                         static_cast<double>(feature_buffer[0][0]));
    return;
  }

  // Obtain the quantized output from model's output tensor
  float y = output->data.f[0];
  
  float max_score;
    int max_index;
    for (int i = 0; i < 4; ++i) {
      if ((i == 0) || (y > max_score)) {
        max_score = y;
        max_index = i;
      }
    }

  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
  HandleOutput(error_reporter, max_index);
}
