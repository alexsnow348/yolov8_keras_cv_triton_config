import onnxruntime as ort
import numpy as np
import tensorflow as tf
import os
import cv2
import json
from line_profiler import LineProfiler
from predict_from_freeze import (
    decode_regression_to_boxes,
    get_anchors,
    class_mapping,
    draw_detection,
    non_max_suppression,
    dist2bbox,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Change shapes and types to match model
profiler = LineProfiler()

def run_inference(sess):
    results_ort = sess.run(
        [sess.get_outputs()[0].name, sess.get_outputs()[1].name], {"input_8": data}
    )
    boxes = results_ort[0]
    scores = results_ort[1]
    boxes = decode_regression_to_boxes(boxes)
    anchor_points, stride_tensor = get_anchors(image_shape=data.shape[1:])
    stride_tensor = tf.expand_dims(stride_tensor, axis=-1)
    box_preds = dist2bbox(boxes, anchor_points) * stride_tensor
    result = non_max_suppression(box_preds, scores)
    return result


profiler.add_function(run_inference)
profiler.enable_by_count()

image_path = (
    "/home/wut/playground/Core_MLOps/cell_counting/data/05_model_input/yolo_annotations/cell_rgb_001_rgb_100_rgb_011_rgb_010_rgb_110_rgb_101_cell cluster/images/Sub A-1 Well P-9_2023_07_17__14_12_19__10.png"
)
img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

data = np.asarray(img, dtype=np.float32).reshape(1, 1265, 1268, 3)
# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# Following code assumes NVIDIA GPU is available, you can specify other execution providers or don't include providers parameter
# to use default CPU provider.
# sess = ort.InferenceSession("dst/path/model.onnx", providers=["CUDAExecutionProvider"])
sess = ort.InferenceSession("keras/quantized_models/onnx/Yolov8_s_400um_combined_small_new.onnx")
print("output name 1 :", sess.get_outputs()[0].name)
print("output name 2  :", sess.get_outputs()[1].name)
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)
result = run_inference(sess)
num_detections = result["num_detections"].numpy()
classes = result["classes"].numpy()
boxes = result["boxes"].numpy()
confidence = result["confidence"].numpy()
to_write = {
    "P_9_2023_07_17__14_12_19__10": {
        "boxes": boxes.tolist(),
        "scores": confidence.tolist(),
    }
}
# write boxes to a json file
with open('data/onnx_boxes_300.json', 'w') as f:
    json.dump(to_write, f)

class_labels = [class_mapping[c] for c in classes]
draw_detection(image_path, boxes, class_labels, draw_path="data/", post_fix="_Yolov8_s_400um_combined_small_0_6_0_3_onnx.jpg")

profiler.print_stats()