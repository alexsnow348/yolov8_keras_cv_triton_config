import os
import tensorflow as tf
import cv2
import numpy as np
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src import bounding_box
from line_profiler import LineProfiler

# customize to use cpu only for onnx conversion
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# CLASS_NAMES = ["RGB_100", "RGB_011", "RGB_001", "RGB_010", "RGB_101", "Object of concern", "RGB_110", "Cell cluster"]
CLASS_NAMES = [
    "cell",
    "rgb_001",
    "rgb_100",
    "rgb_011",
    "rgb_010",
    "rgb_110",
    "rgb_101",
    "cell_cluster",
]
class_mapping = dict(zip(range(len(CLASS_NAMES)), CLASS_NAMES))

BOX_REGRESSION_CHANNELS = 64
SCORE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.3


def decode_regression_to_boxes(preds):
    """Decodes the results of the YOLOV8Detector forward-pass into boxes.

    Returns left / top / right / bottom predictions with respect to anchor
    points.

    Each coordinate is encoded with 16 predicted values. Those predictions are
    softmaxed and multiplied by [0..15] to make predictions. The resulting
    predictions are relative to the stride of an anchor box (and correspondingly
    relative to the scale of the feature map from which the predictions came).
    """
    preds_bbox = keras.layers.Reshape((-1, 4, BOX_REGRESSION_CHANNELS // 4))(preds)
    preds_bbox = ops.nn.softmax(preds_bbox, axis=-1) * ops.arange(
        BOX_REGRESSION_CHANNELS // 4, dtype="float32"
    )
    return ops.sum(preds_bbox, axis=-1)


def get_anchors(
    image_shape,
    strides=[8, 16, 32],
    base_anchors=[0.5, 0.5],
):
    """Gets anchor points for YOLOV8.

    YOLOV8 uses anchor points representing the center of proposed boxes, and
    matches ground truth boxes to anchors based on center points.

    Args:
        image_shape: tuple or list of two integers representing the height and
            width of input images, respectively.
        strides: tuple of list of integers, the size of the strides across the
            image size that should be used to create anchors.
        base_anchors: tuple or list of two integers representing the offset from
            (0,0) to start creating the center of anchor boxes, relative to the
            stride. For example, using the default (0.5, 0.5) creates the first
            anchor box for each stride such that its center is half of a stride
            from the edge of the image.

    Returns:
        A tuple of anchor centerpoints and anchor strides. Multiplying the
        two together will yield the centerpoints in absolute x,y format.

    """
    base_anchors = ops.array(base_anchors, dtype="float32")

    all_anchors = []
    all_strides = []
    for stride in strides:
        hh_centers = ops.arange(0, image_shape[0], stride)
        ww_centers = ops.arange(0, image_shape[1], stride)
        ww_grid, hh_grid = ops.meshgrid(ww_centers, hh_centers)
        grid = ops.cast(
            ops.reshape(ops.stack([hh_grid, ww_grid], 2), [-1, 1, 2]),
            "float32",
        )
        anchors = (
            ops.expand_dims(base_anchors * ops.array([stride, stride], "float32"), 0)
            + grid
        )
        anchors = ops.reshape(anchors, [-1, 2])
        all_anchors.append(anchors)
        all_strides.append(ops.repeat(stride, anchors.shape[0]))

    all_anchors = ops.cast(ops.concatenate(all_anchors, axis=0), "float32")
    all_strides = ops.cast(ops.concatenate(all_strides, axis=0), "float32")

    all_anchors = all_anchors / all_strides[:, None]

    # Swap the x and y coordinates of the anchors.
    all_anchors = ops.concatenate(
        [all_anchors[:, 1, None], all_anchors[:, 0, None]], axis=-1
    )
    return all_anchors, all_strides


def dist2bbox(distance, anchor_points):
    """Decodes distance predictions into xyxy boxes.

    Input left / top / right / bottom predictions are transformed into xyxy box
    predictions based on anchor points.

    The resulting xyxy predictions must be scaled by the stride of their
    corresponding anchor points to yield an absolute xyxy box.
    """
    left_top, right_bottom = ops.split(distance, 2, axis=-1)
    x1y1 = anchor_points - left_top
    x2y2 = anchor_points + right_bottom
    return ops.concatenate((x1y1, x2y2), axis=-1)  # xyxy bbox


def non_max_suppression(
    box_prediction, class_prediction, images=None, image_shape=None
):
    """Accepts images and raw predictions, and returns bounding box
    predictions.

    Args:
        box_prediction: Dense Tensor of shape [batch, boxes, 4] in the
            `bounding_box_format` specified in the constructor.
        class_prediction: Dense Tensor of shape [batch, boxes, num_classes].
    """

    class_prediction = ops.sigmoid(class_prediction)
    confidence_prediction = ops.max(class_prediction, axis=-1)
    box_prediction_numpy = box_prediction[0].numpy()
    scores_numpy = confidence_prediction[0].numpy()

    # numpy to tensor
    box_tensor = ops.array(box_prediction_numpy)
    score_tensor = ops.array(scores_numpy)
    length = len([i for i in scores_numpy if i > SCORE_THRESHOLD])
    selected_indices, _ = tf.image.non_max_suppression_with_scores(
        boxes=box_tensor,
        max_output_size=length,
        score_threshold=SCORE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        scores=score_tensor,
    )

    selected_boxes = tf.gather(box_prediction[0], selected_indices)
    selected_score = tf.gather(confidence_prediction[0], selected_indices)
    class_prediction = tf.gather(class_prediction[0], selected_indices)
    bounding_boxes = {
        "boxes": selected_boxes,
        "confidence": selected_score,
        "classes": ops.argmax(class_prediction, axis=-1),
        "num_detections": ops.array([len(selected_indices)]),
    }

    # this is required to comply with KerasCV bounding box format.
    return bounding_boxes


def load_image(image_path):
    image = tf.io.read_file(image_path)
    return image


def transform_xyxy_to_minmax(x1, y1, x2, y2):
    xmin = min(x1, x2)
    ymin = min(y1, y2)
    xmax = max(x1, x2)
    ymax = max(y1, y2)
    return xmin, ymin, xmax, ymax


def transform_xywh_to_minmax(x, y, w, h):
    xmin = x - w / 2
    ymin = y - h / 2
    xmax = x + w / 2
    ymax = y + h / 2
    return xmin, ymin, xmax, ymax


def draw_detection(img_path, bboxes, class_labels, draw_path="data/"):
    """Draw bounding boxes on the image."""
    image = cv2.imread(img_path)
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        xmin, ymin, xmax, ymax = transform_xyxy_to_minmax(x1, y1, x2, y2)

        # x, y, w, h = bbox
        # xmin, ymin, xmax, ymax = transform_xywh_to_minmax(x, y, w, h)

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
    save_suffix = img_path.split("/")[-1].split(".")[0] + "_with_bbox_more.jpg"
    save_path = draw_path + save_suffix
    cv2.imwrite(save_path, image)


profiler = LineProfiler()
profiler.add_function(non_max_suppression)
profiler.enable_by_count()

if __name__ == "__main__":
    # image_path = "/home/wut/playground/DQ_Arralyze_MachineLearning/rest_related/mldeployment/test/TestImg.png"
    image_path = "/home/haider/playground/test/samples/Sub A-1 Well A-15_2023_07_20__11_34_33__13.png"
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = np.asarray(img, dtype=np.float32).reshape(1, 1265, 1268, 3)
    # reloaded_artifact = tf.saved_model.load("models/yolov8_tf/1/model.savedmodel")
    reloaded_artifact = tf.saved_model.load(
        "/data/models/haider/YOLOv8/yolov8_saved_models/Yolov8_s_400um_combined_small"
    )
    # reloaded_artifact = tf.saved_model.load("/data/models/haider/YOLOv8/yolov8_saved_models/Yolov8_400um_unstained_cell_small")
    print(reloaded_artifact.signatures["serving_default"].structured_input_signature)
    print(reloaded_artifact.signatures["serving_default"].structured_outputs)
    predictions = reloaded_artifact.serve(data)

    boxes = predictions["boxes"]
    scores = predictions["classes"]
    boxes = decode_regression_to_boxes(boxes)
    anchor_points, stride_tensor = get_anchors(image_shape=data.shape[1:])
    stride_tensor = ops.expand_dims(stride_tensor, axis=-1)
    box_preds = dist2bbox(boxes, anchor_points) * stride_tensor
    result = non_max_suppression(box_preds, scores, image_shape=data.shape[1:])
    num_detections = result["num_detections"].numpy()
    classes = result["classes"].numpy()
    boxes = result["boxes"].numpy()
    confidence = result["confidence"].numpy()
    class_labels = [class_mapping[c] for c in classes]
    draw_detection(image_path, boxes, class_labels)
    profiler.print_stats()
