import os
import subprocess
# python -m tf2onnx.convert --saved-model "/data/models/haider/YOLOv8/yolov8_saved_models/Yolov8_s_400um_combined_small" \
#     --opset 18 --output "keras/Yolov8_s_400um_combined_small.onnx"

# customize to use cpu only for onnx conversion
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Pfade zu den Modellen
saved_model_path = "/data/models/haider/YOLOv8/yolov8_saved_models/Yolov8_s_400um_combined_small"
onnx_model_path = "keras/quantized_models/onnx/Yolov8_s_400um_combined_small_new.onnx"
command = [
    "python", "-m", "tf2onnx.convert",
    "--saved-model", saved_model_path,
    "--opset", "13",
    "--output", onnx_model_path
]

subprocess.run(command)


