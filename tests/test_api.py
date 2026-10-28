
# Import TestClient for simulated HTTP requests
from fastapi.testclient import TestClient

# Import FastAPI application from main.py
from main import app

# Create a reusable test client
client = TestClient(app)


def test_health_endpoint():
    """
    Verify that the application is running
    and responding to health-check requests.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_valid_penguin_prediction():
    """
    Verify that valid penguin measurements
    produce a species prediction.
    """
    payload = {
        "culmen_length_mm": 39.1,
        "culmen_depth_mm": 18.7,
        "flipper_length_mm": 181,
        "body_mass_g": 3750
    }

    # Send simulated POST request
    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    # Convert response JSON to dictionary
    result = response.json()

    # Check both fields exist
    assert "prediction" in result
    assert "class_name" in result

    # Check predicted species
    assert result["class_name"] in [
        "Adelie",
        "Chinstrap",
        "Gentoo"
    ]

    # Our model returns species names, not integers
    assert result["prediction"] in [
        "Adelie",
        "Chinstrap",
        "Gentoo"
    ]

    assert result["prediction"] == result["class_name"]


def test_invalid_penguin_input():
    """
    Verify that invalid penguin measurements
    are rejected by FastAPI.
    """
    payload = {
        "culmen_length_mm": -2.3,
        "culmen_depth_mm": 18.7,
        "flipper_length_mm": 181,
        "body_mass_g": 3750
    }

    # Send invalid input
    response = client.post("/predict", json=payload)

    # FastAPI returns 422 for validation errors
    assert response.status_code == 422
