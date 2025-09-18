import json
import logging
import os
import argparse
import torch


logger = logging.getLogger(__name__)

DEFAULT_MAX_PROMPT_TOKENS = 512
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 1.0
DEFAULT_FREQUENCY_PENALTY = 0.0


class Config:
    _instance = None  # Class variable to store the singleton instance

    @classmethod
    def get_instance(cls):
        """Get the global Config instance, creating it if it doesn't exist"""
        if cls._instance is None:
            cls._instance = Config()
        return cls._instance

    def __init__(
        self,
        model_id: str = "gpt2-small",
        custom_hf_model_id: str | None = None,
        model_dtype: str = "float32",
        secret: str | None = None,
        port: int = 5002,
        token_limit: int = 100,
        num_layers: int | None = None,
        device: str | None = None,
        model_from_pretrained_kwargs: str = "{}",
        special_token_ids: list[int] | None = None,
    ):
        self.model_id = model_id
        self.custom_hf_model_id = custom_hf_model_id
        self.model_dtype = model_dtype
        self.secret = secret
        self.port = port
        self.token_limit = token_limit
        self.num_layers = num_layers
        self.device = device
        self.model_kwargs = json.loads(model_from_pretrained_kwargs)
        self.special_token_ids = (
            torch.tensor(special_token_ids).to(self.device)
            if special_token_ids is not None
            else None
        )

        # Log configuration details after initialization
        logger.info(
            f"Initialized Config with:\n"
            f"  model_id: {self.model_id}\n"
            f"  custom_hf_model_id: {self.custom_hf_model_id}\n"
            f"  model_dtype: {self.model_dtype}\n"
            f"  port: {self.port}\n"
            f"  token_limit: {self.token_limit}\n"
            f"  device: {self.device}\n"
            f"  special_token_ids: {self.special_token_ids}\n"
        )

    def set_num_layers(self, num_layers: int) -> None:
        self.num_layers = num_layers

    def set_special_token_ids(self, special_token_ids: list[int] | set[int]) -> None:
        self.special_token_ids = torch.tensor(special_token_ids).to(self.device)


def parse_env_and_args():
    args = argparse.Namespace()

    args.host = os.getenv("HOST", "0.0.0.0")
    args.port = int(os.getenv("PORT", "5002"))
    args.model_id = os.getenv("MODEL_ID", "gpt2-small")
    args.custom_hf_model_id = os.getenv("CUSTOM_HF_MODEL_ID", None)
    args.model_dtype = os.getenv("MODEL_DTYPE", "float32")
    args.token_limit = int(os.getenv("TOKEN_LIMIT", "200"))
    args.model_from_pretrained_kwargs = os.getenv("MODEL_FROM_PRETRAINED_KWARGS", "{}")
    args.sentry_dsn = os.getenv("SENTRY_DSN")

    args.device = os.getenv("DEVICE")
    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    return args
