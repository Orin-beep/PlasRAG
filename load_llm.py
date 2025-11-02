#coding=utf-8
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model(model_id: str, hf_token: str = None):
    """
    Download and load a Hugging Face model (open or gated).

    Args:
        model_id (str): Hugging Face model identifier, e.g.,
                        'Qwen/Qwen3-30B-A3B-Instruct-2507'
        hf_token (str, optional): Hugging Face access token if model is gated.
    """
    print("\n=== LLM Model Loader ===")
    print(f"Selected model: {model_id}")

    if hf_token:
        print("🔐 Using provided Hugging Face token for gated model access.")
    else:
        print("🌐 No token provided — attempting to download an open model.")

    print("\n🔄 Downloading and loading model... This may take a few minutes.\n")

    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )

    print(f"\n✅ Successfully loaded model: {model_id}")
    return tok, model


def main():
    parser = argparse.ArgumentParser(description="Load a Hugging Face LLM (open or gated).")
    parser.add_argument("--model", required=True, help="Hugging Face model name (e.g. Qwen/Qwen3-30B-A3B-Instruct-2507)")
    parser.add_argument("--token", default=None, help="Hugging Face access token for gated models (optional)")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model, args.token)


if __name__ == "__main__":
    main()
