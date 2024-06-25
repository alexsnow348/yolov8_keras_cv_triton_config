import keras_cv
import os

# customize to use cpu only for onnx conversion
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_s_backbone_coco")
class_ids = [
    "cell",
    "rgb_100",
    "rgb_010",
    "rgb_001",
    "rgb_110",
    "rgb_011",
    "rgb_101",
    "cell_cluster",
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

model = keras_cv.models.YOLOV8Detector(
    num_classes=len(class_mapping),
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=3,
)
model.prediction_decoder = keras_cv.layers.MultiClassNonMaxSuppression(
    bounding_box_format="xyxy",
    from_logits=True,
    iou_threshold=0.2,
    confidence_threshold=0.6,
)
model.load_weights("model_yolov8small.h5")

# Create the artifact
model.export("model.savedmodel")  # export as tensorflow saved model
