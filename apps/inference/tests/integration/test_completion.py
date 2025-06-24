from fastapi.testclient import TestClient

from neuronpedia_inference_client.models.steer_completion_request import SteerCompletionRequest
from neuronpedia_inference_client.models.np_steer_method import NPSteerMethod
from neuronpedia_inference_client.models.np_steer_type import NPSteerType
from neuronpedia_inference_client.models.np_steer_feature import NPSteerFeature
from neuronpedia_inference_client.models.steer_completion_post200_response import SteerCompletionPost200Response

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
                strength=1.0,
            )
        ],
        n_completion_tokens=5,
        temperature=0.5,
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

    expected_steered_output = "Hello, world! I'm a developer"
    expected_default_output = "Hello, world! I'm going to be"
    
    assert response_model.outputs[0].output["STEERED"].startswith(expected_steered_output)    
    assert response_model.outputs[1].output["DEFAULT"].startswith(expected_default_output)
