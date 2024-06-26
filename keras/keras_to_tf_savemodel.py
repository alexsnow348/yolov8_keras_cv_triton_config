import keras_cv
import os

# customize to use cpu only for onnx conversion
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_s_backbone_coco")
CLASS_NAMES = ["RGB_100", "RGB_011", "RGB_001", "RGB_010", "RGB_101", "Object of concern", "RGB_110", "Cell cluster"]
class_mapping = dict(zip(range(len(CLASS_NAMES)), CLASS_NAMES))    


model = keras_cv.models.YOLOV8Detector(
    num_classes=len(class_mapping),
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=3,
)
# model.prediction_decoder = keras_cv.layers.MultiClassNonMaxSuppression(
#     bounding_box_format="xyxy",
#     from_logits=True,
#     iou_threshold=0.2,
#     confidence_threshold=0.6,
# )
model.load_weights("keras/model_yolov8small.h5")

# Create the artifact
model.export("models/yolov8_tf/1/model.savedmodel")  # export as tensorflow saved model

