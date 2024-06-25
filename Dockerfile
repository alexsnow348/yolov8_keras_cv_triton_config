FROM nvcr.io/nvidia/tritonserver:23.01-py3

# Install dependencies
RUN pip install opencv-python && \
    pip install --upgrade keras-cv tensorflow && \
    apt update && \
    apt install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

CMD ["tritonserver", "--model-repository=/models" 	]