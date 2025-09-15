# Base image: Triton Server (Python backend có sẵn)
FROM nvcr.io/nvidia/tritonserver:24.05-py3

# Cài thêm HuggingFace + PyTorch
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    transformers==4.41.2 \
    accelerate \
    sentencepiece

# Copy model repository vào container
COPY model_repository /models

# Start Triton khi container khởi động
CMD ["tritonserver", "--model-repository=/models"]
