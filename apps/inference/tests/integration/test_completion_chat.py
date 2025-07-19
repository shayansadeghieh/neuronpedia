from fastapi.testclient import TestClient
from neuronpedia_inference_client.models.np_steer_chat_message import NPSteerChatMessage
from neuronpedia_inference_client.models.np_steer_feature import NPSteerFeature
from neuronpedia_inference_client.models.np_steer_method import NPSteerMethod
from neuronpedia_inference_client.models.np_steer_type import NPSteerType
from neuronpedia_inference_client.models.steer_completion_chat_post200_response import (
    SteerCompletionChatPost200Response,
)
from neuronpedia_inference_client.models.steer_completion_chat_post_request import (
    SteerCompletionChatPostRequest,
)

from tests.conftest import MODEL_ID, SAE_SELECTED_SOURCES, TEST_PROMPT, X_SECRET_KEY

ENDPOINT = "/v1/steer/completion-chat"
N_COMPLETION_TOKENS = 10
TEMPERATURE = 0
STRENGTH = 10.0  # Steering mechanism (feature or vector) specific strength
STRENGTH_MULTIPLIER = 10.0  # Multiplier across all steering mechanisms
STEER_FEATURE_INDEX = 5
SEED = 42
FREQ_PENALTY = 0.0
STEER_SPECIAL_TOKENS = False


def test_completion_chat_steered_with_features_additive(client: TestClient):
    """
    Test steering using features with additive method for chat completion.
    """
    request = SteerCompletionChatPostRequest(
        prompt=[NPSteerChatMessage(content=TEST_PROMPT, role="user")],
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
        steer_special_tokens=STEER_SPECIAL_TOKENS,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    response_model = SteerCompletionChatPost200Response(**data)

    # Create a mapping of output type to output text
    outputs_by_type = {output.type: output.raw for output in response_model.outputs}

    # Test basic API contract
    assert len(outputs_by_type) == 2
    assert NPSteerType.STEERED in outputs_by_type
    assert NPSteerType.DEFAULT in outputs_by_type

    # Both outputs should contain some completion text
    assert len(outputs_by_type[NPSteerType.STEERED]) > 0
    assert len(outputs_by_type[NPSteerType.DEFAULT]) > 0

    # Steered output should be different from default output
    assert outputs_by_type[NPSteerType.STEERED] != outputs_by_type[NPSteerType.DEFAULT]

    expected_steered_output = "<|endoftext|><|im_start|>user\nHello, world!<|im_end|>\n<|im_start|>assistant\n\n<|im_start|>user\n"
    expected_default_output = "<|endoftext|><|im_start|>user\nHello, world!<|im_end|>\n<|im_start|>assistant\n<|im_end|>\n<|"

    assert outputs_by_type[NPSteerType.STEERED] == expected_steered_output
    assert outputs_by_type[NPSteerType.DEFAULT] == expected_default_output
