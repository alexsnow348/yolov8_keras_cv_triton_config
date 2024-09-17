import json
import tensorflow as tf
import numpy as np
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src import bounding_box

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

BOX_REGRESSION_CHANNELS = 64


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        bounding_boxes_config = pb_utils.get_output_config_by_name(
            model_config, "bounding_boxes"
        )
        classes_names_config = pb_utils.get_output_config_by_name(
            model_config, "classes_names"
        )
        confidence_config = pb_utils.get_output_config_by_name(
            model_config, "confidence"
        )


        # Convert Triton types to numpy types
        self.bounding_boxes_dtype = pb_utils.triton_string_to_numpy(
            bounding_boxes_config["data_type"]
        )

        self.classes_names_dtype = pb_utils.triton_string_to_numpy(
            classes_names_config["data_type"]
        )
        self.confidence_dtype = pb_utils.triton_string_to_numpy(
            confidence_config["data_type"]
        )

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        threshold = float(self.model_config["parameters"]["THRESHOLD"]["string_value"])
        iouthreshold = float(
            self.model_config["parameters"]["IOUTHRESHOLD"]["string_value"]
        )
        label_map_path = self.model_config["parameters"]["LABEL_MAP_PATH"][
            "string_value"
        ]
        with open(label_map_path, "r") as f:
            label_map = json.load(f)

        responses = []

        def decode_regression_to_boxes(preds):
            """Decodes the results of the YOLOV8Detector forward-pass into boxes.

            Returns left / top / right / bottom predictions with respect to anchor
            points.

            Each coordinate is encoded with 16 predicted values. Those predictions are
            softmaxed and multiplied by [0..15] to make predictions. The resulting
            predictions are relative to the stride of an anchor box (and correspondingly
            relative to the scale of the feature map from which the predictions came).
            """
            preds_bbox = keras.layers.Reshape((-1, 4, BOX_REGRESSION_CHANNELS // 4))(
                preds
            )
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
                    ops.expand_dims(
                        base_anchors * ops.array([stride, stride], "float32"), 0
                    )
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
            length = len([i for i in scores_numpy if i > threshold])
            selected_indices, _ = tf.image.non_max_suppression_with_scores(
                    boxes=box_tensor,
                    max_output_size=length,
                    score_threshold=threshold,
                    iou_threshold=iouthreshold,
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
        
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            scores = pb_utils.get_input_tensor_by_name(request, "classes")
            boxes = pb_utils.get_input_tensor_by_name(request, "boxes")
            input_image = pb_utils.get_input_tensor_by_name(request, "input_8")
            input_image = input_image.as_numpy()
            image_shape = input_image.shape[1:]
            scores = scores.as_numpy()
            boxes = boxes.as_numpy()
            boxes_tensor = tf.convert_to_tensor(boxes, np.float32)
            scores_tensor = tf.convert_to_tensor(scores, np.float32)
            
            boxes = decode_regression_to_boxes(boxes_tensor)
            anchor_points, stride_tensor = get_anchors(image_shape=image_shape)
            stride_tensor = ops.expand_dims(stride_tensor, axis=-1)

            box_preds = dist2bbox(boxes, anchor_points) * stride_tensor
            box_preds = bounding_box.convert_format(
                box_preds, source="xyxy", target="xyxy",
                image_shape=image_shape
            )
            result = non_max_suppression(box_preds, scores_tensor, 
                                         image_shape=image_shape)

            num_detections = result["num_detections"].numpy()
            # Create a reverse mapping (swap keys and values)
            reverse_mapping_dict = {v: k for k, v in label_map.items()}
            class_list = result["classes"].numpy()
            # Map the provided value to its corresponding key using the reverse mapping
            mapped_label = [reverse_mapping_dict.get(value) for value in class_list]
            bounding_boxes = result["boxes"].numpy()
            confidence = result["confidence"].numpy()

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_bounding_boxes = pb_utils.Tensor(
                "bounding_boxes",
                np.array(bounding_boxes).astype(self.bounding_boxes_dtype),
            )
            out_classes_names = pb_utils.Tensor(
                "classes_names", np.array(mapped_label).astype(self.classes_names_dtype)
            )
            out_confidence = pb_utils.Tensor(
                "confidence", np.array(confidence).astype(self.confidence_dtype)
            )


            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_bounding_boxes, 
                                out_classes_names,
                                out_confidence]
            )
            responses.append(inference_response)
        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses
