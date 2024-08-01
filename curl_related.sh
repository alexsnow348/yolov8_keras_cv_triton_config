# to trigger dummpy triton inference call
curl -X POST http://localhost:8000/v2/models/yolov8_combined_ensemble/versions/1/infer  \
    -H "Content-Type: application/json"  -d @input_data.json

# to enable trace for single inference
curl -X POST http://localhost:8000/v2/models/yolov8_combined_ensemble/trace/setting \
     -H "Content-Type: application/json" \
     -d '{
           "trace_level": ["TIMESTAMPS"],
           "trace_rate": "1",
           "trace_count": "-1",
           "log_frequency": "1"
         }'

# to start triton server
docker run --name=TritonInferenceServer --rm  \
  -p8000:8000 -p8001:8001 -p8002:8002 -v ./models/:/model_repo yolov8-triton:latest \
   tritonserver --model-repository=/model_repo

