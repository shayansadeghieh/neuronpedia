"""
Integration tests for the chat completion endpoint.
These tests use TestClient to start the FastAPI server directly.
"""

import pytest
import json
from neuronpedia_inference.endpoints.dataclasses import SteerMethod
from fastapi.testclient import TestClient

# Import all constants from conftest
from tests.conftest import (
    MODEL_ID,
    EXAMPLE_USER_MESSAGE,
    ACCEPTABLE_SOUNDS,
    EXAMPLE_MESSAGES,
    EXAMPLE_STEER_OPTIONS,
    DOG_VECTOR,
    TEST_SEED,
    EXAMPLE_SEED,
    MAX_TOKENS,
    TEST_TEMPERATURE,
)


class TestChatCompletionIntegration:
    """Integration test class for chat completion endpoint."""

    def test_chat_completion_request_structure_validation(self):
        request_data = {
            "model": MODEL_ID,
            "messages": EXAMPLE_MESSAGES,
            "seed": EXAMPLE_SEED,
            "max_tokens": MAX_TOKENS,
            "stream": True,
            "temperature": TEST_TEMPERATURE,
            "steer_options": EXAMPLE_STEER_OPTIONS,
        }

        # This test just validates the structure is correct
        # The actual server test will verify it works
        assert request_data["model"] == MODEL_ID
        assert len(request_data["messages"]) == 1
        assert request_data["messages"][0]["role"] == "user"
        assert request_data["messages"][0]["content"] == EXAMPLE_USER_MESSAGE
        assert request_data["seed"] == EXAMPLE_SEED
        assert request_data["stream"] is True

        steer_options = request_data["steer_options"]
        assert steer_options["steer_method"] == SteerMethod.SIMPLE_ADDITIVE
        assert steer_options["normalize_steering"] is False
        assert len(steer_options["vectors"]) == 1

        vector = steer_options["vectors"][0]
        assert vector["vector"] == DOG_VECTOR["vector"]
        assert vector["strength"] == DOG_VECTOR["strength"]
        assert vector["hook"] == DOG_VECTOR["hook"]

    def test_server_health_check(self, test_client: TestClient):
        """Test that the server is running and responding to health checks."""
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_chat_completion_streaming_endpoint_exact_example(
        self, test_client: TestClient
    ):
        request_data = {
            "model": MODEL_ID,
            "messages": EXAMPLE_MESSAGES,
            "seed": EXAMPLE_SEED,
            "max_tokens": MAX_TOKENS,
            "stream": True,
            "temperature": TEST_TEMPERATURE,
            "steer_options": EXAMPLE_STEER_OPTIONS,
        }

        # Make the request using TestClient
        response = test_client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json", "Authorization": "Bearer cat"},
        )

        # Check response status and headers
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")

        # Collect streaming response chunks from response content
        chunks = []
        for line in response.content.decode().split("\n"):
            if line and line.startswith("data: "):
                chunks.append(line)

        # Should have at least: first chunk (role), content chunks, final chunk, [DONE]
        assert len(chunks) >= 3

        # Check first chunk (should contain role and prompt tokens)
        first_chunk_data = json.loads(chunks[0][6:])  # Remove "data: " prefix
        assert first_chunk_data["model"] == MODEL_ID
        assert first_chunk_data["object"] == "chat.completion.chunk"
        assert first_chunk_data["choices"][0]["delta"]["role"] == "assistant"
        assert "tokens" in first_chunk_data["choices"][0]["delta"]
        # Check that we get content chunks (exclude first, final, and [DONE])
        content_chunks = [chunk for chunk in chunks[1:-2] if chunk != "data: [DONE]"]
        assert len(content_chunks) >= 0  # May be 0 for very short responses

        # Check that at least one chunk contains the expected content (case insensitive)
        found_acceptable_sounds = False
        for chunk in content_chunks:
            chunk_data = json.loads(chunk[6:])  # Remove "data: " prefix
            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                delta = chunk_data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content.lower().strip() in ACCEPTABLE_SOUNDS:
                    found_acceptable_sounds = True
                    break

        assert (
            found_acceptable_sounds
        ), f"Expected to find '{ACCEPTABLE_SOUNDS}' in at least one content chunk (case insensitive)"

        # Check final chunk has usage information
        final_chunk_data = json.loads(
            chunks[-2][6:]
        )  # Second to last should be final chunk
        assert "usage" in final_chunk_data
        assert final_chunk_data["usage"]["prompt_tokens"] > 0
        assert final_chunk_data["usage"]["completion_tokens"] >= 0

        # Check [DONE] marker
        assert chunks[-1] == "data: [DONE]"

    def test_chat_completion_non_streaming_endpoint_exact_example(
        self, test_client: TestClient
    ):
        """Test the non-streaming chat completion endpoint with the same request but stream=False."""
        # Request data matching example.sh but with stream=False
        request_data = {
            "model": MODEL_ID,
            "messages": EXAMPLE_MESSAGES,
            "seed": EXAMPLE_SEED,
            "max_tokens": MAX_TOKENS,
            "temperature": TEST_TEMPERATURE,
            "stream": False,  # Changed to False for non-streaming
            "steer_options": EXAMPLE_STEER_OPTIONS,
        }

        # Make the request
        response = test_client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json", "Authorization": "Bearer cat"},
        )

        # Check response status and headers
        assert response.status_code == 200
        assert response.headers.get("content-type") == "application/json"

        # Parse JSON response
        response_data = response.json()

        # Validate response structure
        assert response_data["model"] == MODEL_ID
        assert response_data["object"] == "chat.completion"
        assert "id" in response_data
        assert "created" in response_data
        assert len(response_data["choices"]) == 1

        # Check choice structure
        choice = response_data["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert "content" in choice["message"]
        assert choice["finish_reason"] == "stop"

        # Check usage information
        assert "usage" in response_data
        usage = response_data["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] >= 0
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )

    def test_different_steering_methods(self, test_client: TestClient):
        """Test different steering methods work."""
        # Test ORTHOGONAL_DECOMP method with Normalized Steering
        request_data = {
            "model": MODEL_ID,
            "messages": EXAMPLE_MESSAGES,
            "stream": False,
            "temperature": TEST_TEMPERATURE,
            "max_tokens": MAX_TOKENS,  # Keep short for test speed
            "steer_options": {
                "steer_method": SteerMethod.ORTHOGONAL_DECOMP,
                "normalize_steering": True,
                "vectors": [
                    {
                        "vector": DOG_VECTOR["vector"],
                        "strength": DOG_VECTOR["strength"],
                        "hook": DOG_VECTOR["hook"],
                    }
                ],
            },
        }

        response = test_client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json", "Authorization": "Bearer cat"},
        )

        response_data = response.json()

        response_content = response_data["choices"][0]["message"]["content"].lower()
        print(f"Response content: {response_content}")
        contains_acceptable_sound = any(
            sound in response_content for sound in ACCEPTABLE_SOUNDS
        )
        assert contains_acceptable_sound

        assert response.status_code == 200
        assert response_data["model"] == MODEL_ID

    def test_seed_reproducibility(self, test_client: TestClient):
        """Test that the same seed produces consistent results."""
        request_data = {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "Count to 3"}],
            "seed": TEST_SEED,
            "stream": False,
            "max_tokens": MAX_TOKENS,
            "temperature": TEST_TEMPERATURE,
        }

        # Make two identical requests
        response1 = test_client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json", "Authorization": "Bearer cat"},
        )

        response2 = test_client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json", "Authorization": "Bearer cat"},
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        # The responses should be identical due to the same seed
        data1 = response1.json()
        data2 = response2.json()

        # Content should be the same (tokens might differ due to timing, but content should match)
        assert (
            data1["choices"][0]["message"]["content"]
            == data2["choices"][0]["message"]["content"]
        )

    def test_request_without_steering(self, test_client: TestClient):
        """Test that requests work without steering options."""
        request_data = {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "max_tokens": MAX_TOKENS,
            "temperature": TEST_TEMPERATURE,
        }

        response = test_client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json", "Authorization": "Bearer cat"},
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["model"] == MODEL_ID
        assert len(response_data["choices"]) == 1
        assert response_data["choices"][0]["message"]["role"] == "assistant"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
