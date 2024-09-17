from _pytest.monkeypatch import V
import tensorflow as tf
import numpy as np
import keras_cv
import os
import pathlib
import yaml
from tqdm import tqdm
import json
# customize to use cpu only for onnx conversion
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

DATASET_INFO_PATH = "/home/wut/playground/Core_MLOps/cell_counting/data/05_model_input/yolo_annotations/cell_rgb_001_rgb_100_rgb_011_rgb_010_rgb_110_rgb_101_cell cluster/dataset_info.yaml"
TRAIN_DATASET_IMAGE_PATH = "/home/wut/playground/Core_MLOps/cell_counting/data/05_model_input/yolo_annotations/cell_rgb_001_rgb_100_rgb_011_rgb_010_rgb_110_rgb_101_cell cluster/images"
BATCH_SIZE = 1
TRAIN_DATASET_LABEL_PATH = "/home/wut/playground/Core_MLOps/cell_counting/data/05_model_input/yolo_annotations/cell_rgb_001_rgb_100_rgb_011_rgb_010_rgb_110_rgb_101_cell cluster/labels"
VALIDATION_SPLIT_SIZE = 0.2

def get_class_mapping_from_yml(dataset_info_path=DATASET_INFO_PATH):
    """Read class mapping from YAML file"""

    with open(dataset_info_path, "r") as f:
        class_mapping_info = yaml.safe_load(f)
    class_names = class_mapping_info["names"]
    class_mapping = dict(zip(range(len(class_names)), class_names))
    return class_mapping

def load_image(image_path):
    """Load image from file path"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def load_dataset_to_keras_cv_format(image_path, classes, bbox):
    """Load dataset from file path"""
    # Read Image
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    single_data = {
        "images": tf.cast(image, tf.float32),
        "bounding_boxes": bounding_boxes,
    }
    return single_data


def read_annotation_from_label_file(label_file, path_images, class_mapping):
    boxes = []
    classes = []
    image_name = label_file.split("/")[-1].replace(".txt", ".png")
    image_path = os.path.join(path_images, image_name)

    with open(label_file) as f:
        lines = f.readlines()  # list containing lines of file
        for line in lines:
            line_list = list(map(float, line.split()))
            cls = int(line_list[0])  # class id starts from 1, convert to 0-based index
            cls_name = class_mapping[cls]
            # y_min, x_min, y_max_, x_max
            ymin = line_list[1]
            xmin = line_list[2]
            ymax = line_list[3]
            xmax = line_list[4]
            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(cls_name)
    class_ids = [
        list(class_mapping.keys())[list(class_mapping.values()).index(cls)]
        for cls in classes
    ]
    return image_path, boxes, class_ids

def generate_box_and_class_mapping(class_mapping, label_filelist):
    """Generate box and class mapping"""
    image_paths = []
    bbox = []
    classes = []
    for label_file in tqdm(label_filelist):
        image_path, boxes, class_ids = read_annotation_from_label_file(
            label_file, TRAIN_DATASET_IMAGE_PATH, class_mapping
        )
        image_paths.append(image_path)
        bbox.append(boxes)
        classes.append(class_ids)
    return image_paths, bbox, classes


def generate_label_file_list():
    """Generate by getting all TXT file paths and sort them"""
    label_file_list = sorted(
        [
            os.path.join(TRAIN_DATASET_LABEL_PATH, file_name)
            for file_name in os.listdir(TRAIN_DATASET_LABEL_PATH)
            if file_name.endswith(".txt")
        ]
    )
    return label_file_list


def transform_to_tf_ragged_tensor(image_paths, classes, bbox):
    """Transform to tf ragged tensor"""
    image_paths = tf.ragged.constant(image_paths)
    classes = tf.ragged.constant(classes)
    bbox = tf.ragged.constant(bbox)
    ragged_tensor_dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, classes, bbox)
    )
    return ragged_tensor_dataset




class_mapping = get_class_mapping_from_yml()
label_filelist = generate_label_file_list()
print( label_filelist[300:301],)
image_paths, bbox, classes = generate_box_and_class_mapping(
        class_mapping,
        label_filelist[300:301],
    )
to_write = {"image_300": bbox[0]}
# write bbox to a json file
with open('data/bbox_grounth_truth_300.json', 'w') as f:
    json.dump(to_write, f)
ragged_tensor_dataset = transform_to_tf_ragged_tensor(image_paths, classes, bbox)


def representative_dummy_dataset():
    for input_value in ragged_tensor_dataset.take(1):
        # input_value is a tuple of (image_paths, classes, bbox)
        # read image from image_paths
        input_image = load_image(input_value[0])
        input_image = np.expand_dims(input_image, axis=0).astype(np.float32)
        # Model has only one input so each data point has one element.
        yield [input_image]
        
# saved_model_dir = "/data/models/haider/YOLOv8/yolov8_saved_models/Yolov8_s_400um_combined_small"
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.representative_dataset = representative_dummy_dataset
# converter.inference_input_type = tf.uint8  # or tf.uint8
# converter.inference_output_type = tf.uint8  # or tf.uint8
# tflite_quant_model = converter.convert()


# tflite_models_dir = pathlib.Path("keras/quantized_models/tflite/")
# tflite_models_dir.mkdir(exist_ok=True, parents=True)

# # Save the quantized model:
# tflite_model_quant_file = tflite_models_dir/"Yolov8_s_400um_combined_small.tflite"
# tflite_model_quant_file.write_bytes(tflite_quant_model)

for input_value in ragged_tensor_dataset.take(1):
    # input_value is a tuple of (image_paths, classes, bbox)
    # read image from image_paths
    input_image = load_image(input_value[0])
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32)
    # Model has only one input so each data point has one element.
    print(input_image.shape)