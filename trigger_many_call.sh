#!/bin/bash

for ((i=1; i<=1000; i++))
do
    curl -X POST http://localhost:8000/v2/models/yolov8_combined_ensemble/versions/1/infer  \
    -H "Content-Type: application/json"  -d @input_data.json
    sleep 0.2
done