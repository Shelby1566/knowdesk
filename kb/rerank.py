import os
import time
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

class LocalReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_qwen3 = "qwen3-reranker" in model_name.lower()
        if self.is_qwen3:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.causal_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device).eval()
            self.token_yes = self.tokenizer.convert_tokens_to_ids("yes")
            self.token_no = self.tokenizer.convert_tokens_to_ids("no")
        else:
            self.cross_encoder = CrossEncoder(
                model_name,
                device=self.device,
                max_length=512,
                model_kwargs={"torch_dtype": torch.float16} if self.device == "cuda" else {},
            )

    def score_pairs(self, query: str, passages: list[str], batch_size: int = 32) -> list[float]:
        if not passages:
            return []
        q_short = query[:450]
        if not self.is_qwen3:
            pairs = [[q_short, p[:600]] for p in passages]
            scores = self.cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            return [float(s) for s in np.asarray(scores).tolist()]

        prompts = [
            f"<|im_start|>system\nJudge whether the Document meets the requirements based on the Query. "
            f"Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
            f"<|im_start|>user\n<Query>: {q_short}\n<Document>: {p[:600]}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
            for p in passages
        ]
        all_scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                enc = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(self.device)
                logits = self.causal_model(**enc).logits[:, -1, :]
                yes_no = logits[:, [self.token_no, self.token_yes]]
                probs = torch.softmax(yes_no.float(), dim=-1)[:, 1]
                all_scores.extend(probs.cpu().tolist())
        return all_scores

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
        top_k: int = 10,
    ) -> tuple[list[tuple[str, float]], float]:
        if not candidates:
            return [], 0.0
        t0 = time.perf_counter()
        ids = [cid for cid, _ in candidates]
        texts = [txt for _, txt in candidates]
        scores = self.score_pairs(query, texts)
        ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ranked[:top_k], elapsed_ms
