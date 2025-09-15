import os
import json
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForCausalLM

class TritonPythonModel:
    def initialize(self, args):
        # Load config JSON với encoding an toàn (fix BOM)
        model_dir = os.path.dirname(__file__)
        json_path = os.path.join(model_dir, "model.json")
        with open(json_path, "r", encoding="utf-8-sig") as f:
            self.cfg = json.load(f)

        model_name = self.cfg["model"]

        # Chọn device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")

        # Load tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        print(f"[INFO] Model {model_name} loaded successfully")

        # Chuẩn bị generation config từ JSON
        self.gen_kwargs = {
            "max_new_tokens": self.cfg.get("max_new_tokens", 50),
            "temperature": self.cfg.get("temperature", 0.7),
            "top_p": self.cfg.get("top_p", 0.9),
            "do_sample": self.cfg.get("do_sample", True),
            "repetition_penalty": self.cfg.get("repetition_penalty", 1.2),
            "no_repeat_ngram_size": self.cfg.get("no_repeat_ngram_size", 0),
        }
        print(f"[INFO] Generation config: {self.gen_kwargs}")

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                # Lấy input từ Triton
                prompt = pb_utils.get_input_tensor_by_name(request, "PROMPT").as_numpy()[0].decode("utf-8")

                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                # Generate với config từ JSON
                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, **self.gen_kwargs)

                # Decode
                output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

                # Cắt prompt ra, chỉ lấy phần model sinh
                if output_text.startswith(prompt):
                    output_text = output_text[len(prompt):].strip()

                # Output
                out_tensor = pb_utils.Tensor("COMPLETION", np.array([output_text.encode()], dtype=np.bytes_))
                responses.append(pb_utils.InferenceResponse([out_tensor]))

            except Exception as e:
                error_msg = f"[ERROR] {str(e)}"
                out_tensor = pb_utils.Tensor("COMPLETION", np.array([error_msg.encode()], dtype=np.bytes_))
                responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
