FROM nvcr.io/nvidia/tritonserver:24.03-py3

# Install dependencies
RUN pip --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --no-cache-dir install opencv-python && \
    pip --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --no-cache-dir install --upgrade keras-cv tensorflow && \
    apt update && \
    apt install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

CMD ["tritonserver", "--model-repository=/models" 	]