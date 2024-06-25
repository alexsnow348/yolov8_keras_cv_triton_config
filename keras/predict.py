import os
import tensorflow as tf
import keras_cv
import cv2
import numpy as np
from keras_cv import bounding_box
from keras_cv import visualization
from matplotlib import pyplot as plt

# customize to use cpu only for onnx conversion
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
class_ids = ["cell", "rgb_100", "rgb_010", "rgb_001", "rgb_110", "rgb_011", "rgb_101", "cell_cluster"]
class_mapping = dict(zip(range(len(class_ids)), class_ids))    

def load_image(image_path):
    image = tf.io.read_file(image_path)
    return image

def transform_xyxy_to_minmax(x1, y1, x2, y2):
    xmin = min(x1, x2)
    ymin = min(y1, y2)
    xmax = max(x1, x2)
    ymax = max(y1, y2)
    return xmin, ymin, xmax, ymax

def draw_detection(img_path, bboxes, class_labels, draw_path="../data/"):
    """Draw bounding boxes on the image."""
    image = cv2.imread(img_path)
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        xmin, ymin, xmax, ymax = transform_xyxy_to_minmax(x1, y1, x2, y2)
        if class_labels[i].upper() == "RGB_100":
            color = (0, 0, 255)
        elif class_labels[i].upper() == "RGB_010":
            color = (0, 255, 0)
        elif class_labels[i].upper() == "RGB_001":
            color = (255, 0, 0)
        else:
            color = (255, 255, 255)
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        cv2.putText(
            image,
            class_labels[i],
            (int(xmin) - 10, int(ymin) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    save_suffix = img_path.split("/")[-1].split(".")[0] + "_with_bbox_keras.jpg"
    save_path = draw_path + save_suffix

    cv2.imwrite(save_path, image)


if __name__ == "__main__":
    # inference
    backbone = keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_s_backbone_coco"
    )

    model = keras_cv.models.YOLOV8Detector(
        num_classes=len(class_mapping),
        bounding_box_format="xyxy",
        backbone=backbone,
        fpn_depth=3,
    )
    model.load_weights('model_yolov8small.h5')
    # Customizing non-max supression of model prediction. I found these numbers to work fairly well
    model.prediction_decoder = keras_cv.layers.MultiClassNonMaxSuppression(
        bounding_box_format="xyxy",
        from_logits=True,
        iou_threshold=0.2,
        confidence_threshold=0.6,
    )

    image_path = "/home/wut/playground/DQ_Arralyze_MachineLearning/rest_related/mldeployment/test/TestImg.png"
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = np.asarray(img, dtype=np.float32).reshape(1, 1265, 1268, 3)
    result = model.predict(data)
   
    
    num_detections = result["num_detections"][0]
    classes = list(result["classes"][0][:num_detections].numpy())
    boxes = result["boxes"][0][:num_detections].numpy()
    confidence = result["confidence"][0][:num_detections]
    class_labels = [class_mapping[c] for c in classes]
    draw_detection(image_path, boxes, class_labels)
   
  