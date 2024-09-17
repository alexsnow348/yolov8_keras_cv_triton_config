import onnxruntime as ort
import numpy as np
import tensorflow as tf
import os
import cv2
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

def run_inference(predictions):
    boxes = predictions[0]
    scores = predictions[1]
    boxes = decode_regression_to_boxes(boxes)
    anchor_points, stride_tensor = get_anchors(image_shape=data.shape[1:])
    stride_tensor = tf.expand_dims(stride_tensor, axis=-1)
    box_preds = dist2bbox(boxes, anchor_points) * stride_tensor
    result = non_max_suppression(box_preds, scores)
    return result


profiler.add_function(run_inference)
profiler.enable_by_count()

image_path = (
    "/home/wut/playground/yolov8/yolov8_keras_cv_triton_config/data/TestImg.png"
)
img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_image = np.asarray(img, dtype=np.uint8).reshape(1, 1265, 1268, 3)

tflite_model_file = "keras/quantized_models/tflite/Yolov8_s_400um_combined_small.tflite"
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

# Abrufen der Eingabe- und Ausgabedetails
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Überprüfen der Eingabeform
expected_input_shape = input_details[0]['shape']
print('Erwartete Eingabeform: ', expected_input_shape)


# Anpassen der Eingabeform an die erwartete Form
if input_image.shape != tuple(expected_input_shape):
    input_image = np.resize(input_image, expected_input_shape)
    
# Setzen des Eingabetensors
interpreter.set_tensor(input_details[0]['index'], input_image)
    
# Ausführen des Interpreters
interpreter.invoke()
    
# Abrufen der Ausgabe
output = interpreter.get_tensor(output_details[0]['index'])
predictions = output[0]
result = run_inference(predictions)
num_detections = result["num_detections"].numpy()
classes = result["classes"].numpy()
boxes = result["boxes"].numpy()
confidence = result["confidence"].numpy()
class_labels = [class_mapping[c] for c in classes]
draw_detection(image_path, boxes, class_labels, draw_path="data/", post_fix="_Yolov8_s_400um_combined_small.jpg")

profiler.print_stats()