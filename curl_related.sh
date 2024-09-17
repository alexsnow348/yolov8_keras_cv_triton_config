# YOLO
# to trigger dummpy triton inference call
curl -X POST http://localhost:8000/v2/models/yolov8_combined_ensemble/versions/1/infer  \
    -H "Content-Type: application/json"  -d @input_data_yolo.json

curl -X POST http://localhost:8000/v2/models/yolov8_combined_ensemble_onnx/versions/1/infer  \
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
 
  
## Faster RCNN
# to trigger dummpy triton inference call
curl -X POST http://localhost:8000/v2/models/ensemble_model/versions/1/infer  \
    -H "Content-Type: application/json"  -d @input_data_fastercnn.json

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
docker run --name=TritonInferenceServer --rm --shm-size=2g   \
  -p8000:8000 -p8001:8001 -p8002:8002 -v  ./models:/model_repo yolov8-triton:latest \
  tritonserver --model-repository=/model_repo --trace-config mode=opentelemetry --trace-config opentelemetry,url=http://10.10.12.30:4318/v1/traces \
  --trace-config triton,log-frequency=1,rate=1,count=1,level=TIMESTAMPS

# sudo mount -t cifs -o user=testuser,domain=testdomain //192.168.1.100/freigabe /mnt

sudo mount -t cifs -o username=wut.hlaing,domain=LPKF.com //10.10.10.45/ARRALYZE_Picture /data/biolab_arralyze_picture_data

./bin/opensearch-plugin  install https://github.com/aiven/prometheus-exporter-plugin-for-opensearch/releases/download/2.13.0.0/prometheus-exporter-2.13.0.0.zip

# python -m tf2onnx.convert --input frozen_east_text_detection.pb --inputs "input_images:0" --outputs "feature_fusion/Conv_7/Sigmoid:0","feature_fusion/concat_3:0" --output detection.onnx
model-analyzer profile --model-repository ./models \
--profile-models yolov8_combined_ensemble_onnx --triton-launch-mode=local \
--output-model-repository-path data/model_analyzer \
-f perf.yaml --override-output-model-repository --latency-budget 10 --run-config-search-mode quick


model-analyzer report --report-model-configs yolov8_combined_ensemble_onnx,yolov8_combined_ensemble \
 --export-path data/model_analyzer  

model-analyzer profile -m ./models --profile-models  yolov8_combined_ensemble_onnx,yolov8_combined_ensemble --cpu-only-composing-models yolov8_combined_ensemble_onnx,yolov8_combined_ensemble 

model-analyzer profile \
    --model-repository ./models \
    --profile-models add_sub --triton-launch-mode=docker \
    --output-model-repository-path data/model_analyzer \
    --export-path profile_results