# Báo cáo về việc Triển khai OPT-125M trên NVIDIA Triton Inference Server

## Nguyễn Huy Hoàng

## 1. Yêu cầu hệ thống
- **GPU**: NVIDIA RTX 4060 Laptop GPU (8GB VRAM).
- **Docker Desktop** (có NVIDIA Container Toolkit).
- **Python 3.10+** trên host (dùng để chạy client test).

---

## 2. Cấu trúc thư mục
```
model_repository/
└── vllm_model
    ├── 1
    │   ├── model.py
    │   └── model.json
    └── config.pbtxt
Dockerfile
client_http.py
```

---

## 3. File cấu hình

### `config.pbtxt`
```protobuf
name: "vllm_model"
backend: "python"

input [
  {
    name: "PROMPT"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]

output [
  {
    name: "COMPLETION"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]
```

### `model.json`
```json
{
  "model": "facebook/opt-125m",
  "disable_log_requests": "true",
  "gpu_memory_utilization": 0.7,
  "max_num_batched_tokens": 2048,
  "max_num_seqs": 32,
  "do_sample": true,
  "temperature": 0.7,
  "top_p": 0.9,
  "max_new_tokens": 64,
  "repetition_penalty": 1.2,
  "no_repeat_ngram_size": 3
}
```

---

## 4. Dockerfile
```dockerfile
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
```

---

## 5. Build image
```powershell
docker build -t triton-opt125m .
```

---

## 6. Chạy Triton Server
```powershell
docker run --gpus all -it --rm `
  -v "C:/Users/huyho/Downloads/model_repository:/models:rw" `
  -p8000:8000 -p8001:8001 -p8002:8002 `
  --name triton_opt125m_mount triton-opt125m
```

Đã load model thành công và mô hình sẵn sàng để test:
```
+------------+---------+--------+
| Model      | Version | Status |
+------------+---------+--------+
| vllm_model | 1       | READY  |
+------------+---------+--------+
```

---

## 7. Client test (`client_http.py`)
```python
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

URL = "localhost:8000"
MODEL_NAME = "vllm_model"

def infer_prompts(prompts, timeout=60):
    client = httpclient.InferenceServerClient(url=URL)
    if not client.is_model_ready(MODEL_NAME):
        raise RuntimeError(f"Model {MODEL_NAME} not ready")

    arr = np.array([p.encode("utf-8") for p in prompts], dtype=object)
    infer_input = httpclient.InferInput("PROMPT", [len(prompts)], "BYTES")
    infer_input.set_data_from_numpy(arr)

    outputs = [httpclient.InferRequestedOutput("COMPLETION")]

    try:
        result = client.infer(MODEL_NAME, [infer_input], outputs, timeout=timeout)
        return [x.decode("utf-8") for x in result.as_numpy("COMPLETION")]
    except InferenceServerException as e:
        print("Triton inference error:", e)
        return None

if __name__ == "__main__":
    prompts = ["Hello, how are you?"]
    outs = infer_prompts(prompts)
    if outs:
        print("Model output:", outs[0])
```

---

## 8. Kết quả sau khi chạy
```powershell
python client_http.py
```

Output:
```
Model output: I'm doing okay. I was just really busy with school and getting my foot in the door.  But now that I'm back, I feel like I can handle it even if I don't get paid! So thank you for asking! :)
```

---

## 9. Ghi chú sau khi infer xong:
- OPT-125M khá nhỏ, output chưa tự nhiên hoặc có thể dẫn đến lạc đề, nhưng sau cùng thì deploy được mô hình. 
- Nếu cần chất lượng cao hơn 
- Có thể tinh chỉnh hyperparameters trong `model.json` (`temperature`, `top_p`, `repetition_penalty`, …).
