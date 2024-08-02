import cv2
import json
import requests
import kserve
import numpy as np
import tritonclient.http as httpclient

im = cv2.imread(
    "/home/MLDeployment/yolov8_keras_cv_triton_config/data/TestImg.png"
)
api_url = "http://localhost:8000/v2"
response = requests.get(api_url)
response.content

with open("server_info.json", "w") as f:
    # Write the response to a JSON file
    json.dump(response.json(), f)
    
try:
    # Need to specify large enough concurrency to issue all the
    # inference requests to the server in parallel.
    triton_client = httpclient.InferenceServerClient(
        url="localhost:8000", verbose=False, concurrency=100
    )
except Exception as e:
    print("context creation failed: " + str(e))
model_name = "yolov8_ensemble"
dat = np.asarray(im, dtype=np.uint8).reshape(1, 1265, 1268, 3)
inferInput = kserve.InferInput(
    "input_tensor", [1, 1265, 1268, 3], datatype="UINT8", data=dat
)
inferRequest = kserve.InferRequest(infer_inputs=[inferInput], model_name="yolov8_ensemble")
restRequest = inferRequest.to_rest()
r = requests.post(
    "http://localhost:8000/v2/models/ensemble_model/versions/1/infer",
    json=restRequest,
)
# r = requests.post(
#     "http://mlserver1.lpkf.com:8000/v2/models/yolov8_tf/versions/1/infer",
#     json=restRequest,
# )
# r.json()
with open("infer_yolov8_ensemble_with_confidence_test.json", "w") as f:
    # Write the response to a JSON file
    json.dump(r.json(), f)