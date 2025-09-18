import asyncio
import gc
import json
import logging
import os
import traceback
from collections.abc import Awaitable
from typing import Callable

import sentry_sdk
import torch
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuronpedia_inference.config import Config, parse_env_and_args
from neuronpedia_inference.endpoints.chat_completion import (
    router as chat_completion_router,
)
from neuronpedia_inference.logging import initialize_logging
from neuronpedia_inference.shared import STR_TO_DTYPE, Model
from neuronpedia_inference.utils import checkCudaError

initialize_logging()

logger = logging.getLogger(__name__)
logger.info("Server module initialized")

load_dotenv()

global initialized
initialized = False

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

args = parse_env_and_args()


# we have to initialize model AFTER server startup, because some infrastructure providers require
# our server to respond to health checks within a few minutes of starting up
@app.on_event("startup")  # pyright: ignore[reportDeprecated]
async def startup_event():
    logger.info("Starting initialization...")
    # Wait briefly to ensure server is ready
    await asyncio.sleep(3)
    # Start initialization in background
    asyncio.create_task(initialize(args.custom_hf_model_id))
    logger.info("Initialization started")


v1_router = APIRouter(prefix="/v1")

v1_router.include_router(chat_completion_router)

app.include_router(v1_router)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/initialize")
async def initialize(
    custom_hf_model_id: str | None = None,
):
    logger.info("Initializing...")

    # Move the heavy operations to a separate thread pool to prevent blocking
    def load_model():

        # clear cuda cache and gc in case of previous initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.set_grad_enabled(False)
        checkCudaError("cpu")

        # todo: use multiple devices if available
        device_count = 1

        logger.info(f"Args: {args}")

        SECRET = os.getenv("SECRET")

        config = Config(
            secret=SECRET,
            model_id=args.model_id,
            port=args.port,
            custom_hf_model_id=custom_hf_model_id,
            model_dtype=args.model_dtype,
            token_limit=args.token_limit,
            device=args.device,
            model_from_pretrained_kwargs=args.model_from_pretrained_kwargs,
        )
        Config._instance = config

        logger.info("Loading model...")

        hf_model = None
        hf_tokenizer = None
        if custom_hf_model_id is not None:
            logger.info("Loading custom HF model: %s", custom_hf_model_id)
            hf_model = AutoModelForCausalLM.from_pretrained(
                custom_hf_model_id,
                torch_dtype=STR_TO_DTYPE[config.model_dtype],
            )
            hf_tokenizer = AutoTokenizer.from_pretrained(custom_hf_model_id)

        model = HookedTransformer.from_pretrained_no_processing(
            config.model_id,
            device=args.device,
            dtype=STR_TO_DTYPE[config.model_dtype],
            n_devices=device_count,
            hf_model=hf_model,
            **({"hf_config": hf_model.config} if hf_model else {}),
            tokenizer=hf_tokenizer,
            **config.model_kwargs,
        )

        # Print model information
        logger.info(f"Model configuration:")
        logger.info(f"  Model ID: {config.model_id}")
        logger.info(f"  Device: {model.cfg.device}")
        logger.info(f"  Number of layers: {model.cfg.n_layers}")
        logger.info(f"  Number of heads: {model.cfg.n_heads}")
        logger.info(f"  Head dimension: {model.cfg.d_head}")
        logger.info(f"  Model dimension: {model.cfg.d_model}")
        logger.info(f"  Vocabulary size: {model.cfg.d_vocab}")
        logger.info(f"  Context length: {model.cfg.n_ctx}")
        logger.info(f"  Model dtype: {config.model_dtype}")
        if model.tokenizer:
            logger.info(f"  Tokenizer: {type(model.tokenizer).__name__}")
            logger.info(f"  Special token IDs: {model.tokenizer.all_special_ids}")

        # add hook_in to mlp for transcoders
        def add_hook_in_to_mlp(mlp):  # type: ignore
            mlp.hook_in = HookPoint()
            original_forward = mlp.forward
            mlp.forward = lambda x: original_forward(mlp.hook_in(x))

        for block in model.blocks:
            add_hook_in_to_mlp(block.mlp)
        model.setup()

        Model._instance = model
        config.set_num_layers(model.cfg.n_layers)

        if model.tokenizer:
            # save these for later use
            config.set_special_token_ids(model.tokenizer.all_special_ids)

        logger.info(f"Loaded {config.model_id} on {args.device}")
        checkCudaError()

        global initialized
        initialized = True
        logger.info("Initialized: %s", initialized)

    await asyncio.get_event_loop().run_in_executor(None, load_model)


@app.middleware("http")
async def check_secret_key(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    if request.url.path == "/health":
        return await call_next(request)
    config = Config.get_instance()

    # no secret required, just return
    if config.secret is None:
        return await call_next(request)

    # Check Authorization Bearer header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        secret_key = auth_header[7:]  # Remove "Bearer " prefix

    if not secret_key or secret_key != config.secret:
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid or missing Authorization Bearer token"},
        )
    return await call_next(request)


@app.middleware("http")
async def check_model(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    config = Config.get_instance()

    if request.method == "POST":
        try:
            body = await request.json()
            if "model" in body and (
                body["model"] != config.model_id
                and body["model"] != config.custom_hf_model_id
            ):
                logger.error("Unsupported model: %s", body["model"])
                return JSONResponse(
                    content={"error": "Unsupported model"}, status_code=400
                )
        except (json.JSONDecodeError, ValueError):
            pass

    return await call_next(request)


@app.middleware("http")
async def log_and_check_cuda_error(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    if not initialized:
        return JSONResponse(
            status_code=500,
            content={"error": "Server not initialized"},
        )
    logger.info("=== Request Info ===")
    logger.info(f"URL: {request.url}")

    try:
        body = await request.body()
        if body:
            logger.info(f"Body: {body.decode()}")
    except Exception as e:
        logger.error(f"Error reading body: {str(e)}")

    return await call_next(request)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):  # noqa: ARG001
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "traceback": traceback.format_exc() if app.debug else None,
        },
    )


def main():
    if os.getenv("SENTRY_DSN"):
        logger.info("Initializing Sentry")
        sentry_sdk.init(
            dsn=os.getenv("SENTRY_DSN"),
            traces_sample_rate=1.0,
            _experiments={
                "continuous_profiling_auto_start": True,
            },
        )
    else:
        logger.info("SENTRY_DSN not set, skipping Sentry initialization")

    return app
