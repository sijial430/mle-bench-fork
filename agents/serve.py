from __future__ import annotations

"""Lightweight server for the ArmoRM reward model.

Run with one dedicated GPU, e.g.

    CUDA_VISIBLE_DEVICES=0 python -m src.open_r1.serve --model RLHFlow/ArmoRM-Llama3-8B-v0.1 --port 8002

The server exposes two endpoints:
  GET  /health/         → {"status": "ok"}
  POST /score/          → {"scores": [float, ...]}

The request payload must be

    {
        "prompts": [str | list],
        "completions": [str | list]
    }

where each element matches the interface expected by `reward_server_remote`.
"""

import argparse
import os
from typing import Any, List

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import uvicorn

from open_r1.utils import ensure_deepspeed_gather_attr

class ScoreRequest(BaseModel):
    prompts: List[Any]
    completions: List[Any]

class ScoreResponse(BaseModel):
    scores: List[float]

class GenerateRequest(BaseModel):
    prompts: List[Any]
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9

class GenerateResponse(BaseModel):
    completions: List[str]


def load_model(model_id: str, request_type: str, local_files_only: bool = False):
    if request_type == "score":
        print(f"[serve_{request_type}] Loading {model_id}…", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only or bool(os.environ.get("HF_HUB_OFFLINE")),
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            # device_map="auto",
            device_map={"": 0},
            local_files_only=local_files_only or bool(os.environ.get("HF_HUB_OFFLINE")),
        )
    elif request_type == "generate":
        print(f"[serve_{request_type}] Loading {model_id}…", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only or bool(os.environ.get("HF_HUB_OFFLINE")),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            local_files_only=local_files_only or bool(os.environ.get("HF_HUB_OFFLINE")),
        )
        model = ensure_deepspeed_gather_attr(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print(f"[serve_{request_type}] Model loaded – ready to serve requests", flush=True)
    return tokenizer, model


def build_conversation_for_rm(prompt: Any, completion: Any):
    """Return a list of chat messages compatible with `apply_chat_template`."""
    if isinstance(prompt, list):
        # Already in chat-style format; append completion message.
        if isinstance(completion, list):
            return prompt + completion
        return prompt + [{"role": "assistant", "content": completion}]
    else:
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        
def build_conversation_for_ref(prompt: Any):
    """Return a list of chat messages compatible with `apply_chat_template`.

    If `prompt` is already a list[dict], assume chat format and return as is.
    Otherwise treat it as a single user message.
    """
    if isinstance(prompt, list):
        # Assume already formatted for chat generation
        return prompt
    return [{"role": "user", "content": prompt}]


def create_app(tokenizer, model, request_type: str):
    app = FastAPI()

    @app.get("/health/")
    def health():
        return {"status": "ok"}

    @app.post("/score/", response_model=ScoreResponse)
    def score(req: ScoreRequest):
        assert len(req.prompts) == len(req.completions), "prompts and completions must be same length"
        conversations = [build_conversation_for_rm(p, c) for p, c in zip(req.prompts, req.completions)]
        # Convert to plain strings via the tokenizer's chat template
        texts = [tokenizer.apply_chat_template(conv, tokenize=False) for conv in conversations]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            # model outputs `.logits` of shape (B, 1)
            logits = outputs.logits.squeeze(-1)
            if logits.ndim == 0:
                scores = [logits.item()]
            else:
                scores = logits.float().cpu().tolist()
            
        return {"scores": scores}

    # Register generate endpoint only if generation is enabled (but harmless to expose anyway)
    @app.post("/generate/", response_model=GenerateResponse)
    def generate(req: GenerateRequest):
        assert len(req.prompts) > 0, "prompts list must not be empty"

        conversations = [build_conversation_for_ref(p) for p in req.prompts]
        # Convert conversations to a single input string via chat template
        inputs_text = [
            tokenizer.apply_chat_template(conv, tokenize=False) for conv in conversations
        ]

        batch = tokenizer(
            inputs_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        batch = {k: v.to(model.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=req.temperature > 0.0,
            )

        # Extract only the newly generated tokens for each example
        completions: List[str] = []
        for i, output_ids in enumerate(outputs):
            input_len = batch["input_ids"][i].size(0)
            generated_ids = output_ids[input_len:]
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
            completions.append(completion.strip())

        return {"completions": completions}

    return app


def main():
    parser = argparse.ArgumentParser(description="Serve reward model via FastAPI")
    parser.add_argument("--model", default="RLHFlow/ArmoRM-Llama3-8B-v0.1", help="HF model ID or local path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", default=8002, type=int, help="Port to bind; using a non-8000 port to avoid conflicts with vLLM")
    parser.add_argument("--request_type", default="score", choices=["score", "generate"], help="Request type")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Require the model to be present locally (no downloads)",
    )
    args = parser.parse_args()
    
    # default to offline mode and load the model from the local cache
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tokenizer, model = load_model(args.model, args.request_type, args.local_files_only)
    app = create_app(tokenizer, model, args.request_type)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main() 