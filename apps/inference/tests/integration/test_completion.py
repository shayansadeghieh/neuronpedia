import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from neuronpedia_inference_client.models.np_steer_feature import NPSteerFeature
from neuronpedia_inference_client.models.np_steer_method import NPSteerMethod
from neuronpedia_inference_client.models.np_steer_type import NPSteerType
from neuronpedia_inference_client.models.np_steer_vector import NPSteerVector
from neuronpedia_inference_client.models.steer_completion_post200_response import (
    SteerCompletionPost200Response,
)
from neuronpedia_inference_client.models.steer_completion_request import (
    SteerCompletionRequest,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.conftest import MODEL_ID, SAE_SELECTED_SOURCES, TEST_PROMPT, X_SECRET_KEY

ENDPOINT = "/v1/steer/completion"
N_COMPLETION_TOKENS = 10
TEMPERATURE = 0
STRENGTH = 10.0  # Steering mechanism (feature or vector) specific strength
STRENGTH_MULTIPLIER = 10.0  # Multiplier across all steering mechanisms
STEER_FEATURE_INDEX = 5
SEED = 42
FREQ_PENALTY = 0.0


def test_completion_steered_with_features_additive(client: TestClient):
    """
    Test steering using features with additive method.
    """
    request = SteerCompletionRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        features=[
            NPSteerFeature(
                model=MODEL_ID,
                source=SAE_SELECTED_SOURCES[0],
                index=STEER_FEATURE_INDEX,
                strength=STRENGTH,
            )
        ],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    response_model = SteerCompletionPost200Response(**data)

    # Create a mapping of output type to output text
    outputs_by_type = {output.type: output.output for output in response_model.outputs}

    # Test basic API contract
    assert len(outputs_by_type) == 2
    assert NPSteerType.STEERED in outputs_by_type
    assert NPSteerType.DEFAULT in outputs_by_type

    # Both outputs should start with the original prompt
    assert outputs_by_type[NPSteerType.STEERED].startswith(TEST_PROMPT)
    assert outputs_by_type[NPSteerType.DEFAULT].startswith(TEST_PROMPT)

    # Both outputs should be longer than the prompt (completion was generated)
    assert len(outputs_by_type[NPSteerType.STEERED]) > len(TEST_PROMPT)
    assert len(outputs_by_type[NPSteerType.DEFAULT]) > len(TEST_PROMPT)

    # Steered output should be different from default output
    assert outputs_by_type[NPSteerType.STEERED] != outputs_by_type[NPSteerType.DEFAULT]

    expected_steered_output = "Hello, world! the world, the world, the world, the"
    expected_default_output = "Hello, world!\n\nI'm a programmer and I'm a"

    assert outputs_by_type[NPSteerType.STEERED] == expected_steered_output
    assert outputs_by_type[NPSteerType.DEFAULT] == expected_default_output


def test_completion_steered_with_vectors_additive(client: TestClient):
    """
    Test steering using vectors with additive method.
    """
    request = SteerCompletionRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        vectors=[
            NPSteerVector(
                steering_vector=[1000.0]
                * 768,  # We utilize a large vector to ensure the steering vector is impactful
                strength=STRENGTH,
                hook="blocks.7.hook_resid_post",
            )
        ],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 200

    data = response.json()
    response_model = SteerCompletionPost200Response(**data)

    # Create a mapping of output type to output text
    outputs_by_type = {output.type: output.output for output in response_model.outputs}

    # Test basic API contract
    assert len(outputs_by_type) == 2
    assert NPSteerType.STEERED in outputs_by_type
    assert NPSteerType.DEFAULT in outputs_by_type

    # Both outputs should start with the original prompt
    assert outputs_by_type[NPSteerType.STEERED].startswith(TEST_PROMPT)
    assert outputs_by_type[NPSteerType.DEFAULT].startswith(TEST_PROMPT)

    # Both outputs should be longer than the prompt (completion was generated)
    assert len(outputs_by_type[NPSteerType.STEERED]) > len(TEST_PROMPT)
    assert len(outputs_by_type[NPSteerType.DEFAULT]) > len(TEST_PROMPT)

    # Steered output should be different from default output
    assert outputs_by_type[NPSteerType.STEERED] != outputs_by_type[NPSteerType.DEFAULT]

    expected_steered_output = "Hello, world!!!!!!!!!!!"
    expected_default_output = "Hello, world!\n\nI'm a programmer and I'm a"

    assert outputs_by_type[NPSteerType.STEERED] == expected_steered_output
    assert outputs_by_type[NPSteerType.DEFAULT] == expected_default_output


def test_completion_steered_token_limit_exceeded(client: TestClient):
    """
    Test handling of a prompt that exceeds the token limit.
    """
    long_prompt = "This is a test prompt. " * 1000
    request = SteerCompletionRequest(
        prompt=long_prompt,
        model=MODEL_ID,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED],
        features=[
            NPSteerFeature(
                model=MODEL_ID,
                source=SAE_SELECTED_SOURCES[0],
                index=0,
                strength=STRENGTH,
            )
        ],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 400

    data = response.json()
    expected_error_message = "Text too long: 6001 tokens, max is 500"
    assert data["error"] == expected_error_message


def test_completion_steered_with_features_orthogonal(client: TestClient):
    """
    Test steering using features with orthogonal decomposition method.
    """
    request = SteerCompletionRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        steer_method=NPSteerMethod.ORTHOGONAL_DECOMP,
        normalize_steering=False,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        features=[
            NPSteerFeature(
                model=MODEL_ID,
                source=SAE_SELECTED_SOURCES[0],
                index=STEER_FEATURE_INDEX,
                strength=STRENGTH,
            )
        ],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    response_model = SteerCompletionPost200Response(**data)

    # Create a mapping of output type to output text
    outputs_by_type = {output.type: output.output for output in response_model.outputs}

    # Test basic API contract
    assert len(outputs_by_type) == 2
    assert NPSteerType.STEERED in outputs_by_type
    assert NPSteerType.DEFAULT in outputs_by_type

    # Both outputs should start with the original prompt
    assert outputs_by_type[NPSteerType.STEERED].startswith(TEST_PROMPT)
    assert outputs_by_type[NPSteerType.DEFAULT].startswith(TEST_PROMPT)

    # Both outputs should be longer than the prompt (completion was generated)
    assert len(outputs_by_type[NPSteerType.STEERED]) > len(TEST_PROMPT)
    assert len(outputs_by_type[NPSteerType.DEFAULT]) > len(TEST_PROMPT)

    # Steered output should be different from default output
    assert outputs_by_type[NPSteerType.STEERED] != outputs_by_type[NPSteerType.DEFAULT]

    expected_steered_output = "Hello, world! Hy Hy Hy Hy Hy Hy Hy Hy Hy Hy"
    expected_default_output = "Hello, world!\n\nI'm a programmer and I'm a"

    assert outputs_by_type[NPSteerType.STEERED] == expected_steered_output
    assert outputs_by_type[NPSteerType.DEFAULT] == expected_default_output


def test_completion_steered_with_vectors_orthogonal(client: TestClient):
    """
    Test steering using vectors with orthogonal decomposition.
    """
    request = SteerCompletionRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        steer_method=NPSteerMethod.ORTHOGONAL_DECOMP,
        normalize_steering=False,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        vectors=[
            NPSteerVector(
                steering_vector=[1000.0]
                * 768,  # We utilize a large vector to ensure the steering vector is impactful
                strength=STRENGTH,
                hook="blocks.7.hook_resid_post",
            )
        ],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 200

    data = response.json()
    response_model = SteerCompletionPost200Response(**data)

    # Create a mapping of output type to output text
    outputs_by_type = {output.type: output.output for output in response_model.outputs}

    # Test basic API contract
    assert len(outputs_by_type) == 2
    assert NPSteerType.STEERED in outputs_by_type
    assert NPSteerType.DEFAULT in outputs_by_type

    # Both outputs should start with the original prompt
    assert outputs_by_type[NPSteerType.STEERED].startswith(TEST_PROMPT)
    assert outputs_by_type[NPSteerType.DEFAULT].startswith(TEST_PROMPT)

    # Both outputs should be longer than the prompt (completion was generated)
    assert len(outputs_by_type[NPSteerType.STEERED]) > len(TEST_PROMPT)
    assert len(outputs_by_type[NPSteerType.DEFAULT]) > len(TEST_PROMPT)

    # Steered output should be different from default output
    assert outputs_by_type[NPSteerType.STEERED] != outputs_by_type[NPSteerType.DEFAULT]

    expected_steered_output = "Hello, world!!!!!!!!!!!"
    expected_default_output = "Hello, world!\n\nI'm a programmer and I'm a"

    assert outputs_by_type[NPSteerType.STEERED] == expected_steered_output
    assert outputs_by_type[NPSteerType.DEFAULT] == expected_default_output


def test_completion_logprobs(client: TestClient):
    """Test that logprobs are returned for both STEERED and DEFAULT types."""
    request = SteerCompletionRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        features=[
            NPSteerFeature(
                model=MODEL_ID,
                source=SAE_SELECTED_SOURCES[0],
                index=STEER_FEATURE_INDEX,
                strength=STRENGTH,
            )
        ],
        n_completion_tokens=5,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
        n_logprobs=2,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )

    assert response.status_code == 200

    data = response.json()
    response_model = SteerCompletionPost200Response(**data)

    assert len(response_model.outputs) == 2

    # find steered and default outputs
    steered_output = None
    default_output = None
    for output in response_model.outputs:
        if output.type == NPSteerType.STEERED:
            steered_output = output
        elif output.type == NPSteerType.DEFAULT:
            default_output = output

    assert steered_output is not None
    assert default_output is not None

    # both should have logprobs
    for output in [steered_output, default_output]:
        assert (
            output.logprobs is not None
        ), f"logprobs should not be None for {output.type}"
        assert (
            len(output.logprobs) > 0
        ), f"logprobs should contain items for {output.type}"

        for logprob_item in output.logprobs:
            assert logprob_item.token is not None
            assert logprob_item.logprob is not None
            assert logprob_item.top_logprobs is not None
            assert len(logprob_item.top_logprobs) == 2  # Should match n_logprobs

    steered_logprobs = steered_output.logprobs
    default_logprobs = default_output.logprobs

    # get pyright checks to pass
    assert steered_logprobs is not None, "Steered logprobs should not be None"
    assert default_logprobs is not None, "Default logprobs should not be None"
    assert len(steered_logprobs) > 0, "Steered logprobs should not be empty"
    assert len(default_logprobs) > 0, "Default logprobs should not be empty"

    assert (
        steered_logprobs[0].token == " the"
    ), f"Expected steered token ' the', got '{steered_logprobs[0].token}'"
    assert steered_logprobs[0].logprob == pytest.approx(
        -3.33203125, abs=0.02
    ), f"Steered first token logprob: {steered_logprobs[0].logprob}"

    # verify tokens and logprobs have expected structure
    assert len(default_logprobs[0].token) > 0, "Default token should not be empty"
    assert (
        -50 < default_logprobs[0].logprob < 0
    ), f"Default logprob should be reasonable: {default_logprobs[0].logprob}"

    # verify steered and default logprobs are different
    assert (
        steered_logprobs[0].logprob != default_logprobs[0].logprob
    ), "Steered and default logprobs should differ"


@pytest.mark.parametrize(
    "prompt, n_tokens",
    [
        ("The cat sat", 3),
        ("Hello", 1),
    ],
)
def test_completion_logprobs_match_hugging_face(
    client: TestClient, prompt: str, n_tokens: int
):
    """
    Compare the API's returned logprobs (on ITS generated tokens) with Hugging Face
    computed logprobs for the same tokens. This avoids assuming a specific continuation.
    """
    HF_MODEL_ID = "gpt2"
    API_MODEL_ID = "gpt2-small"

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_fast=True)
    tokenizer_kwargs = dict(return_tensors="pt", add_special_tokens=False)
    hf_model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID, torch_dtype=torch.float32
    )
    hf_model.eval()

    request = SteerCompletionRequest(
        prompt=prompt,
        model=API_MODEL_ID,
        n_completion_tokens=n_tokens,
        n_logprobs=1,
        temperature=0,
        types=[
            NPSteerType.STEERED,
            NPSteerType.DEFAULT,
        ],
        features=[
            NPSteerFeature(
                model=API_MODEL_ID,
                source=SAE_SELECTED_SOURCES[0],
                index=0,
                strength=0.0,  # no steering effect
            )
        ],
        freq_penalty=0.0,
        seed=42,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        strength_multiplier=0.0,
    )

    response = client.post(
        ENDPOINT,
        json=request.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert (
        response.status_code == 200
    ), f"API call failed: {response.status_code} - {response.text}"
    data = response.json()

    default_output = next((o for o in data["outputs"] if o["type"] == "DEFAULT"), None)
    assert default_output is not None, "DEFAULT output not found"

    str_tokens = [t["token"] for t in default_output["logprobs"]]
    logprobs = [float(t["logprob"]) for t in default_output["logprobs"]]
    assert len(str_tokens) == n_tokens
    assert len(logprobs) == n_tokens

    text = "".join(str_tokens)
    prompt_ids = tokenizer(prompt, **tokenizer_kwargs)["input_ids"][0]
    combined_ids = tokenizer(prompt + text, **tokenizer_kwargs)["input_ids"][0]
    ids = combined_ids[len(prompt_ids) :]
    assert len(ids) == len(logprobs), "length mismatch after retokenizing API text"

    # HF logprobs at the same positions
    with torch.no_grad():
        logits = hf_model(combined_ids.unsqueeze(0)).logits[0]  # [seq, vocab]
        hf_logprobs = torch.log_softmax(logits, dim=-1)

    start = len(prompt_ids)
    hf_reference_logprobs = [
        hf_logprobs[start + i - 1, tid].item() for i, tid in enumerate(ids)
    ]

    # numerical check, allowing for implementation differences between TransformerLens and HuggingFace
    assert np.allclose(
        logprobs, hf_reference_logprobs, rtol=0.1, atol=0.6
    ), f"logprob mismatch.\nAPI: {logprobs}\nHF:  {hf_reference_logprobs}"


def test_completion_logprobs_streaming(client: TestClient):
    """Test that logprobs work correctly in streaming mode."""
    prompt = "The cat sat"
    n_tokens = 3
    n_logprobs = 2

    request = SteerCompletionRequest(
        prompt=prompt,
        model=MODEL_ID,
        n_completion_tokens=n_tokens,
        n_logprobs=n_logprobs,
        temperature=0,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        features=[
            NPSteerFeature(
                model=MODEL_ID,
                source=SAE_SELECTED_SOURCES[0],
                index=0,
                strength=0.0,
            )
        ],
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        strength_multiplier=0.0,
        freq_penalty=0.0,
        seed=42,
        stream=True,
    )

    response = client.post(
        ENDPOINT,
        json=request.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    chunks = response.content.decode().strip().split("\n\n")
    final_chunk = None
    for chunk in reversed(chunks):
        if chunk.startswith("data: ") and not chunk.endswith("[DONE]"):
            final_chunk = chunk[6:]  # remove 'data: ' prefix
            break

    assert final_chunk is not None, "No valid streaming chunk found"
    resp = json.loads(final_chunk)

    # verify response structure
    assert "outputs" in resp
    assert len(resp["outputs"]) == 2

    # check both outputs have logprobs
    for output in resp["outputs"]:
        assert output["type"] in ["STEERED", "DEFAULT"]
        assert "logprobs" in output
        assert output["logprobs"] is not None
        assert (
            len(output["logprobs"]) > 0
        ), f"Expected logprobs, got {len(output['logprobs'])}"

        # verify logprobs structure
        for logprob_item in output["logprobs"]:
            assert "token" in logprob_item
            assert "logprob" in logprob_item
            assert "top_logprobs" in logprob_item
            assert isinstance(logprob_item["token"], str)
            assert isinstance(logprob_item["logprob"], (int, float))
            assert len(logprob_item["top_logprobs"]) == n_logprobs

            # verify top_logprobs structure
            for top_logprob in logprob_item["top_logprobs"]:
                assert "token" in top_logprob
                assert "logprob" in top_logprob
                assert isinstance(top_logprob["token"], str)
                assert isinstance(top_logprob["logprob"], (int, float))
