# YOLO
# to trigger dummpy triton inference call
curl -X POST http://localhost:8000/v2/models/yolov8_combined_ensemble/versions/1/infer  \
    -H "Content-Type: application/json"  -d @input_data_yolo.json

# to enable trace for single inference
curl -X POST http://localhost:8000/v2/models/yolov8_combined_ensemble/trace/setting \
     -H "Content-Type: application/json" \
     -d '{
           "trace_level": ["TIMESTAMPS"],
           "trace_rate": "1",
           "trace_count": "-1",
           "log_frequency": "1"
         }'

# to start triton server on cpu - yolo
docker run --name=TritonInferenceServer --rm --shm-size=2g  \
  -p8000:8000 -p8001:8001 -p8002:8002 -v ./models/:/model_repo yolov8-triton:latest \
  tritonserver --model-repository=/model_repo --trace-config mode=opentelemetry --trace-config opentelemetry,url=http://10.10.12.30:4318/v1/traces

## Faster RCNN
# to trigger dummpy triton inference call
curl -X POST http://localhost:8000/v2/models/ensemble_model/versions/1/infer  \
    -H "Content-Type: application/json"  -d @input_data_fastercnn.json

# to enable trace for single inference
curl -X POST http://localhost:8000/v2/models/ensemble_model/trace/setting \
     -H "Content-Type: application/json" \
     -d '{
           "trace_level": ["TIMESTAMPS"],
           "trace_rate": "1",
           "trace_count": "-1",
           "log_frequency": "1"
         }'

# to start triton server on cpu - yolo
docker run --name=TritonInferenceServer --rm --shm-size=2g  \
  -p8000:8000 -p8001:8001 -p8002:8002 -v  /data/model_repository:/model_repo yolov8-triton:latest \
  tritonserver --model-repository=/model_repo --trace-config mode=opentelemetry --trace-config opentelemetry,url=http://10.10.12.30:4318/v1/traces

