# Overview
This is a Triton Inference Server container for YOLOv8 models which built with latest keras-cv.

# Directory Structure
```
models/
    yolov8_tf/
        1/
           model.savedmodel
        config.pbtxt
        
    postprocess/
        1/
            model.py
        config.pbtxt
        
    yolov8_ensemble/
        1/
            <Empty Directory>
        config.pbtxt
README.md
```


# Quick Start

1. Build the Docker Container for Triton Inference:
```
DOCKER_NAME="yolov8-triton"
docker build -t $DOCKER_NAME .
```

2. Run Triton Inference Server:
```
DOCKER_NAME="yolov8-triton"
docker run --gpus all \
    -it --rm \
    --net=host \
    --shm-size=2g \
    -v ./models:/models \
    $DOCKER_NAME
```



