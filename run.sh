#!/bin/bash
# source env/bin/activate
python train.py
# python evaluate.py


bazel build -c opt \
  --config=android_arm64 \
  //tensorflow/lite/tools/evaluation/tasks/inference_diff:run_eval

adb push bazel-bin/tensorflow/lite/tools/evaluation/tasks/inference_diff/run_eval /data/local/tmp

adb shell /data/local/tmp/run_eval \
  --model_file=/data/local/tmp/pwc_net.tflite \
  --delegate=gpu

adb shell /data/local/tmp/run_eval \
  --model_file=/data/local/tmp/pwc_net.tflite \
  --delegate=gpu