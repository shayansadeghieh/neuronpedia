"""
Unit tests for the chat completion endpoint.
These tests use mocks and run quickly without starting the actual server.
"""

import pytest

from neuronpedia_inference.endpoints.dataclasses import (
    ChatMessage,
    ChatCompletionRequest,
    SteerOptions,
    SteerVector,
    SteerMethod,
)

# Import constants from conftest
from tests.conftest import (
    MODEL_ID,
    DOG_VECTOR,
    EXAMPLE_SEED,
    TEST_TEMPERATURE_2,
    TEST_OTHER_HOOK,
)

# Local test constants
TEST_MESSAGE = ChatMessage(role="user", content="What are you?")


class TestChatCompletionUnit:
    """Unit test class for chat completion endpoint (fast, mocked tests)."""

    def test_chat_completion_request_validation(self):
        """Test that the request structure matches the example.sh format and validates correctly."""
        # This matches exactly the request from example.sh
        request_data = {
            "model": MODEL_ID,
            "messages": [{"role": TEST_MESSAGE.role, "content": TEST_MESSAGE.content}],
            "seed": EXAMPLE_SEED,
            "stream": True,
            "steer_options": {
                "steer_method": "SIMPLE_ADDITIVE",
                "normalize_steering": False,
                "vectors": [
                    DOG_VECTOR,
                ],
            },
        }

        # Validate that the request can be parsed correctly by Pydantic
        request = ChatCompletionRequest(**request_data)

        assert request.model == MODEL_ID
        assert len(request.messages) == 1
        assert request.messages[0].role == TEST_MESSAGE.role
        assert request.messages[0].content == TEST_MESSAGE.content
        assert request.seed == EXAMPLE_SEED
        assert request.stream is True
        assert request.steer_options is not None
        assert request.steer_options.steer_method == SteerMethod.SIMPLE_ADDITIVE
        assert request.steer_options.normalize_steering is False
        assert len(request.steer_options.vectors) == 1

        vector = request.steer_options.vectors[0]
        assert vector.vector == DOG_VECTOR["vector"]
        assert vector.strength == DOG_VECTOR["strength"]
        assert vector.hook == DOG_VECTOR["hook"]

    def test_steer_options_validation(self):
        """Test that steering options are properly validated."""
        # Test valid steering options
        steer_options = SteerOptions(
            steer_method=SteerMethod.SIMPLE_ADDITIVE,
            normalize_steering=False,
            vectors=[
                SteerVector(
                    vector=DOG_VECTOR["vector"],
                    strength=DOG_VECTOR["strength"],
                    hook=DOG_VECTOR["hook"],
                )
            ],
        )

        assert steer_options.steer_method == SteerMethod.SIMPLE_ADDITIVE
        assert steer_options.normalize_steering is False
        assert len(steer_options.vectors) == 1
        assert steer_options.vectors[0].vector == DOG_VECTOR["vector"]
        assert steer_options.vectors[0].strength == DOG_VECTOR["strength"]
        assert steer_options.vectors[0].hook == DOG_VECTOR["hook"]

    def test_steer_options_orthogonal_decomp(self):
        """Test orthogonal decomposition steering method validation."""
        steer_options = SteerOptions(
            steer_method=SteerMethod.ORTHOGONAL_DECOMP,
            normalize_steering=True,
            vectors=[
                SteerVector(
                    vector=DOG_VECTOR["vector"],
                    strength=DOG_VECTOR["strength"],
                    hook=DOG_VECTOR["hook"],
                )
            ],
        )

        assert steer_options.steer_method == SteerMethod.ORTHOGONAL_DECOMP
        assert steer_options.normalize_steering is True

    def test_chat_message_validation(self):
        """Test that chat messages are properly validated."""
        message = TEST_MESSAGE

        assert message.role == "user"
        assert message.content == "What are you?"
        assert message.name is None
        assert message.tool_calls is None
        assert message.tool_call_id is None

    def test_multiple_steering_vectors(self):
        """Test that multiple steering vectors can be specified."""
        steer_options = SteerOptions(
            steer_method=SteerMethod.SIMPLE_ADDITIVE,
            normalize_steering=False,
            vectors=[
                SteerVector(
                    vector=DOG_VECTOR["vector"],
                    strength=DOG_VECTOR["strength"],
                    hook=DOG_VECTOR["hook"],
                ),
                SteerVector(
                    vector=DOG_VECTOR["vector"],
                    strength=150,
                    hook=TEST_OTHER_HOOK,
                ),
            ],
        )

        assert len(steer_options.vectors) == 2
        assert steer_options.vectors[0].strength == DOG_VECTOR["strength"]
        assert steer_options.vectors[1].strength == 150
        assert steer_options.vectors[0].hook == DOG_VECTOR["hook"]
        assert steer_options.vectors[1].hook == TEST_OTHER_HOOK

    def test_request_defaults(self):
        """Test that request defaults are applied correctly."""
        minimal_request = {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "Hi"}],
        }

        request = ChatCompletionRequest(**minimal_request)

        # Check defaults
        assert request.stream is False  # Default should be False
        assert request.seed is None
        assert request.steer_options is None
        assert request.temperature == 1.0  # Default from Field
        assert request.frequency_penalty == 0.0
        assert request.top_p == 1.0

    def test_request_parameter_validation(self):
        """Test that request parameters are properly validated."""
        # Test temperature bounds
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="test",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=-1.0,  # Below minimum
            )

        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="test",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=3.0,  # Above maximum
            )

        # Test valid temperature
        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=TEST_TEMPERATURE_2,
        )
        assert request.temperature == TEST_TEMPERATURE_2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
