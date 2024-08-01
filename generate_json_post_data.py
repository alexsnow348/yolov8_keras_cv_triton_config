import numpy as np
import json
import cv2

im = cv2.imread(
    "./data/Sub A-1 Well A-15_2023_07_20__11_34_33__13_with_bbox_more.jpg"
)
# Generate dummy input data
# input_data = np.random.rand(1, 1265, 1268, 3).astype(np.float32)
input_data = np.asarray(im, dtype=np.float32).reshape(1, 1265, 1268, 3)

# Flatten the numpy array
flat_input_data = input_data.flatten().tolist()

# Create the JSON payload
payload = {
    "inputs": [
        {
            "name": "input_2",
            "shape": input_data.shape,
            "datatype": "FP32",
            "data": flat_input_data
        }
    ]
}

# Convert the payload to a JSON string
json_payload = json.dumps(payload)

# Save the JSON payload to a file
with open('input_data.json', 'w') as f:
    f.write(json_payload)