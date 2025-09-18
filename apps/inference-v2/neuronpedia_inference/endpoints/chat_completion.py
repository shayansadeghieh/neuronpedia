import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from neuronpedia_inference.config import (
    Config,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_FREQUENCY_PENALTY,
)
from neuronpedia_inference.endpoints.dataclasses import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionUsage,
    SteerMethod,
    SteerVector,
)
from neuronpedia_inference.endpoints.steer_utils import OrthogonalProjector
from neuronpedia_inference.shared import Model, with_request_lock

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat/completions")
@with_request_lock()
async def chat_completions(request: ChatCompletionRequest):
    """
    Create a chat completion, supporting both streaming and non-streaming responses.

    This endpoint mimics the OpenAI Chat Completions API and generates responses
    using the loaded transformer model.
    """
    try:
        completion_id = f"chatcmpl-{int(time.time() * 1000000)}"
        created = int(time.time())

        if request.stream:
            return StreamingResponse(
                create_streaming_response(request, completion_id, created),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/plain; charset=utf-8",
                },
            )

        # Non-streaming response - accumulate streaming results
        accumulated_content = ""
        accumulated_tokens = []
        prompt_tokens = 0
        completion_tokens = 0
        first_chunk = True

        async for chunk_data in create_streaming_response(
            request, completion_id, created
        ):
            if chunk_data.strip() == "data: [DONE]":
                break

            if chunk_data.startswith("data: "):
                chunk_json = json.loads(chunk_data[6:])  # Remove "data: " prefix
                chunk = ChatCompletionChunk(**chunk_json)

                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta

                    # First chunk contains all prompt tokens and role
                    if first_chunk and delta.role == "assistant" and delta.tokens:
                        accumulated_tokens.extend(delta.tokens)
                        first_chunk = False
                    # Subsequent chunks contain generated content and tokens
                    elif not first_chunk:
                        if delta.content:
                            accumulated_content += delta.content
                        if delta.tokens:
                            accumulated_tokens.extend(delta.tokens)

                # Get usage info from the final chunk
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=accumulated_content,
                        tokens=accumulated_tokens,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500,
        )


async def create_streaming_response(
    request: ChatCompletionRequest,
    completion_id: str,
    created: int,
) -> AsyncGenerator[str, None]:
    """Generate streaming response chunks."""
    model = Model.get_instance()
    config = Config.get_instance()

    # we're not training, so we don't need gradients
    torch.set_grad_enabled(False)

    messages = request.messages

    if model.tokenizer is None:
        raise ValueError("Tokenizer is not initialized")

    # Apply chat template to get tokens
    if (
        not hasattr(model.tokenizer, "chat_template")
        or model.tokenizer.chat_template is None
    ):
        raise ValueError("Model's tokenizer does not support chat templates.")
    else:
        messages_tokenized = model.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    messages_tokenized = torch.tensor(messages_tokenized)

    # Check token limit
    if len(messages_tokenized) > DEFAULT_MAX_PROMPT_TOKENS:
        logger.error(
            "Text too long: %s tokens, max is %s",
            len(messages_tokenized),
            DEFAULT_MAX_PROMPT_TOKENS,
        )
        raise ValueError(
            f"Text too long: {len(messages_tokenized)} tokens, max is {DEFAULT_MAX_PROMPT_TOKENS}"
        )

    model.reset_hooks()

    if request.seed is not None:
        torch.manual_seed(request.seed)

    def steering_hook(activations: torch.Tensor, hook: Any) -> torch.Tensor:
        logger.info(f"Steering hook called with activations: {activations.shape}")
        # make a mask for what tokens to steer
        mask = torch.ones(activations.shape[1], device=activations.device)
        if not request.steer_options.steer_special_tokens:
            # Get the current tokens for this batch
            current_tokens = messages_tokenized.to(activations.device)

            # Get all special token IDs and find their indices
            special_token_ids = config.special_token_ids
            for special_token_id in special_token_ids:
                special_indices = (current_tokens == special_token_id).nonzero(
                    as_tuple=True
                )[0]
                mask[special_indices] = 0

        # Apply steering with the mask
        for vector in request.steer_options.vectors:
            steering_vector = torch.tensor(vector.vector).to(activations.device)

            if not torch.isfinite(steering_vector).all():
                raise ValueError("Steering vector contains inf or nan values")

            if request.steer_options.normalize_steering:
                print(f"Normalizing steering vector: {steering_vector.shape}")
                norm = torch.norm(steering_vector)
                if norm == 0:
                    raise ValueError("Zero norm steering vector")
                steering_vector = steering_vector / norm

            # If it's attention hook, reshape it to (n_heads, head_dim)
            if isinstance(vector, SteerVector) and "attn.hook_z" in vector.hook:
                print(f"Reshaping steering vector: {steering_vector.shape}")
                n_heads = model.cfg.n_heads
                d_head = model.cfg.d_head
                steering_vector = steering_vector.view(n_heads, d_head)

            if request.steer_options.steer_method == SteerMethod.SIMPLE_ADDITIVE:
                print(f"Adding steering vector: {steering_vector.shape}")
                activations += vector.strength * steering_vector * mask.unsqueeze(-1)

            elif request.steer_options.steer_method == SteerMethod.ORTHOGONAL_DECOMP:
                print(f"Projecting steering vector: {steering_vector.shape}")
                projector = OrthogonalProjector(steering_vector)
                projected = projector.project(activations, vector.strength)
                activations = activations * (
                    1 - mask.unsqueeze(-1)
                ) + projected * mask.unsqueeze(-1)

        return activations

    editing_hooks = []
    if request.steer_options is not None:
        editing_hooks = [
            (
                vector.hook,
                steering_hook,
            )
            for vector in request.steer_options.vectors
        ]

    n_prompt_tokens = len(messages_tokenized)
    n_completion_tokens = 0

    # First chunk with role and prompt tokens
    first_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(
                    role="assistant",
                    tokens=[
                        model.to_string([token])
                        for token in messages_tokenized.tolist()
                    ],
                ),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    n_max_new_tokens = request.max_tokens or DEFAULT_MAX_TOKENS

    with model.hooks(fwd_hooks=editing_hooks):
        for i, (result) in enumerate(
            model.generate_stream(
                max_tokens_per_yield=1,
                stop_at_eos=True,
                input=messages_tokenized.unsqueeze(0),
                do_sample=True,
                temperature=request.temperature or DEFAULT_TEMPERATURE,
                freq_penalty=request.frequency_penalty or DEFAULT_FREQUENCY_PENALTY,
                top_p=request.top_p or None,
                max_new_tokens=n_max_new_tokens,
            )
        ):
            # "content" is what the user sees, filtering out special tokens.
            # "tokens" is the tokenized version, including special tokens.
            if i == 0:
                # if it's the first iteration, it will return all prompt tokens, which we don't want.
                # so we only use the last token
                content = model.to_string(result[0][-1:])
                tokens = [model.to_string([token]) for token in result[0].tolist()]
                tokens = tokens[-1:]
            else:
                # if it's in the special token ids, don't include it in the content
                special_tokens_mask = ~torch.isin(result[0], config.special_token_ids)
                content = model.to_string(result[0][special_tokens_mask])
                # don't filter out special tokens for the raw token strings
                tokens = [model.to_string([token]) for token in result[0].tolist()]

            n_completion_tokens += 1

            # Content chunk
            content_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=content, tokens=tokens),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {content_chunk.model_dump_json()}\n\n"

    # Final chunk with finish_reason
    final_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(),
                finish_reason="stop",
            )
        ],
    )

    final_chunk.usage = CompletionUsage(
        prompt_tokens=n_prompt_tokens,
        completion_tokens=n_completion_tokens,
        total_tokens=n_prompt_tokens + n_completion_tokens,
    )

    yield f"data: {final_chunk.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"
