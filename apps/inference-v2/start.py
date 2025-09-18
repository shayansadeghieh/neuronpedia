import argparse
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(
        description="Initialize server configuration for Neuronpedia Inference Server."
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5002,
        help="Port number for the server to listen on",
    )
    parser.add_argument(
        "--model_id",
        default="google/gemma-2-2b-it",
        help="The ID of the base model to use (e.g., 'google/gemma-2-2b-it')",
    )
    parser.add_argument(
        "--custom_hf_model_id",
        default=None,
        help=(
            "Optional: Use a custom HF model ID that is not directly supported by TransformerLens.",
            "This is used to run the deepseek-ai/DeepSeek-R1-Distill-Llama-8B model.",
        ),
    )
    parser.add_argument(
        "--model_dtype",
        default="float32",
        help="Data type for model computations",
    )
    parser.add_argument(
        "--token_limit",
        type=int,
        default=200,
        help="Maximum number of tokens to process",
    )
    parser.add_argument(
        "--device",
        help="Device to run the model on",
    )
    parser.add_argument(
        "--model_from_pretrained_kwargs",
        default="{}",
        help="JSON string of additional keyword arguments",
    )
    # Uvicorn specific arguments
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--reload-dir",
        default="neuronpedia_inference",
        help="Directory to watch for changes when reload is enabled",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Only set environment variables if they don't already exist
    if "MODEL_ID" not in os.environ:
        os.environ["MODEL_ID"] = args.model_id
    if "MODEL_DTYPE" not in os.environ:
        os.environ["MODEL_DTYPE"] = args.model_dtype
    if "TOKEN_LIMIT" not in os.environ:
        os.environ["TOKEN_LIMIT"] = str(args.token_limit)
    if "DEVICE" not in os.environ and args.device is not None:
        os.environ["DEVICE"] = args.device
    if "MODEL_FROM_PRETRAINED_KWARGS" not in os.environ:
        os.environ["MODEL_FROM_PRETRAINED_KWARGS"] = args.model_from_pretrained_kwargs
    if "CUSTOM_HF_MODEL_ID" not in os.environ and args.custom_hf_model_id is not None:
        os.environ["CUSTOM_HF_MODEL_ID"] = str(args.custom_hf_model_id)

    uvicorn_args = [
        "uvicorn",
        "neuronpedia_inference.server:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    if args.reload:
        uvicorn_args.extend(["--reload"])
        if args.reload_dir:
            uvicorn_args.extend(["--reload-dir", args.reload_dir])

    subprocess.run(uvicorn_args)


if __name__ == "__main__":
    main()
