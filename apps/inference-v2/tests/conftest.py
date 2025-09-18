"""
Pytest configuration and shared fixtures for integration tests.
"""

import pytest
import asyncio
import os
import gc
from typing import Generator
from fastapi.testclient import TestClient
from tests.data.gemma_2_2b_it_dog_vector import DOG_VECTOR
import torch
from neuronpedia_inference.endpoints.dataclasses import SteerMethod

# Test configuration constants
MODEL_ID = "google/gemma-2-2b-it"
MODEL_DTYPE = "bfloat16"
TOKEN_LIMIT = "64"
DEVICE = "cpu"
MAX_HEALTH_CHECK_ATTEMPTS = 30
HEALTH_CHECK_TIMEOUT = 2
SERVER_INIT_WAIT_TIME = 5

# Test request constants
EXAMPLE_USER_MESSAGE = "Make a sound. Answer in one word."
ACCEPTABLE_SOUNDS = ["bark", "woof", "dog"]
EXAMPLE_MESSAGES = [{"role": "user", "content": EXAMPLE_USER_MESSAGE}]
EXAMPLE_SEED = 16
MAX_TOKENS = 6
TEST_SEED = 42
TEST_TEMPERATURE = 0


# Test timeout constants
MAX_WAIT_TIME = 5 * 60  # 5 minutes
INITIALIZATION_POLL_INTERVAL = 5  # seconds
REQUEST_TIMEOUT = 60  # seconds
SHORT_REQUEST_TIMEOUT = 30  # seconds
QUICK_TEST_TIMEOUT = 10  # seconds

# Test temperature
TEST_TEMPERATURE_2 = 0.7
TEST_OTHER_HOOK = "blocks.15.hook_resid_pre"

# Example steer options for testing
EXAMPLE_STEER_OPTIONS = {
    "steer_method": SteerMethod.SIMPLE_ADDITIVE,
    "normalize_steering": False,
    "vectors": [
        DOG_VECTOR,
    ],
}


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def initialize_models():
    """
    Defining the global state of the app with a session-scoped fixture that initializes the model.

    This fixture will be run once per test session and will be available to all tests
    that need an initialized model. It uses the same initialization logic as the
    server startup.
    """
    # Set environment variables for testing
    os.environ.update(
        {
            "MODEL_ID": MODEL_ID,
            "MODEL_DTYPE": MODEL_DTYPE,
            "TOKEN_LIMIT": TOKEN_LIMIT,
            # "DEVICE": DEVICE,
        }
    )

    # Import server module and re-parse args after setting environment variables
    # This is important to refresh the module-level args in the server module
    from neuronpedia_inference.config import Config
    from neuronpedia_inference.shared import Model
    from neuronpedia_inference.server import initialize

    # Initialize the model
    asyncio.run(initialize())

    yield

    # Cleanup
    Config._instance = None
    Model._instance = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


@pytest.fixture(scope="session")
def test_client(
    initialize_models: None,
) -> Generator[TestClient, None, None]:  # noqa: ARG001
    """
    Create a TestClient with pre-initialized models.
    """
    from neuronpedia_inference.server import app

    client = TestClient(app)
    yield client
