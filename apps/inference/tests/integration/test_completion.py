from fastapi.testclient import TestClient
from neuronpedia_inference_client.models.np_steer_feature import NPSteerFeature
from neuronpedia_inference_client.models.np_steer_method import NPSteerMethod
from neuronpedia_inference_client.models.np_steer_type import NPSteerType
from neuronpedia_inference_client.models.steer_completion_post200_response import (
    SteerCompletionPost200Response,
)
from neuronpedia_inference_client.models.steer_completion_request import (
    SteerCompletionRequest,
)

from tests.conftest import MODEL_ID, SAE_SELECTED_SOURCES, TEST_PROMPT, X_SECRET_KEY

ENDPOINT = "/v1/steer/completion"


def test_completion_basic(client: TestClient):
    # Build request
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
                index=0,
                strength=10.0,
            )
        ],
        n_completion_tokens=10,
        temperature=0,
        strength_multiplier=1.0,
        freq_penalty=0.0,
        seed=42,
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

    assert outputs_by_type[NPSteerType.STEERED] != outputs_by_type[NPSteerType.DEFAULT]
