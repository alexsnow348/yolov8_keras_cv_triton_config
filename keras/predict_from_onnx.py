import onnxruntime as ort
import numpy as np
import tensorflow as tf
import os
import cv2
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Change shapes and types to match model

image_path = "/home/wut/playground/Core_MLOps/cell_counting/data/05_model_input/yolo_annotations/cell_rgb_001_rgb_100_rgb_011_rgb_010_rgb_110_rgb_101_cell cluster/images/Sub A-1 Well A-15_2023_07_20__11_34_33__13.png"
img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
input1 = np.asarray(img, dtype=np.float32).reshape(1, 1265, 1268, 3)
# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# Following code assumes NVIDIA GPU is available, you can specify other execution providers or don't include providers parameter
# to use default CPU provider.
# sess = ort.InferenceSession("dst/path/model.onnx", providers=["CUDAExecutionProvider"])
sess = ort.InferenceSession("keras/quantized_models/onnx/Yolov8_s_400um_combined_small.onnx")
print("output name 1 :", sess.get_outputs()[0].name)
print("output name 2  :", sess.get_outputs()[1].name)
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)
# Set first argument of sess.run to None to use all model outputs in default order
# Input/output names are printed by the CLI and can be set with --rename-inputs and --rename-outputs
# If using the python API, names are determined from function arg names or TensorSpec names.
results_ort = sess.run([ sess.get_outputs()[0].name, sess.get_outputs()[1].name], {"input_8": input1})

# model = tf.saved_model.load("models/yolov8_cell/1/model.savedmodel")
reloaded_artifact = tf.saved_model.load("/data/models/haider/YOLOv8/yolov8_saved_models/Yolov8_s_400um_combined_small")
print(reloaded_artifact.signatures["serving_default"].structured_input_signature) 
print(reloaded_artifact.signatures["serving_default"].structured_outputs)  
predictions = reloaded_artifact.serve(input1)

# print("Results ORT: ", results_ort)
# print("Results TF: ", predictions)
boxes = results_ort[0]
# EagerTensor to list
print("Boxes shape: ", boxes.shape)
print("Box Type: ", type(boxes))
scores = results_ort[1]
# write boxes to a json file
with open('data/onnx_boxes.json', 'w') as f:
    json.dump(boxes, f)

tf_boxes = predictions["boxes"]
tf_scores = predictions["classes"]

# write tf_boxes to a json file
with open('data/tf_boxes.json', 'w') as f:
    json.dump(tf_boxes, f)

#compare the results of the two models with numpy.allclose
print("Boxes equal: ", np.allclose(boxes, tf_boxes, rtol=1e-5, atol=1e-5))
print("Scores equal: ", np.allclose(scores, tf_scores, rtol=1e-5, atol=1e-5))
