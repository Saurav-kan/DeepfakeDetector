# DeepfakeDetector
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/saurav-kan/DeepfakeDetector)

This repository contains a lightweight API for detecting deepfakes in images. The service is built with FastAPI and utilizes a fine-tuned EfficientNet-B0 model for binary classification (Real vs. Fake). The application is containerized with Docker and includes a CI/CD pipeline for automated deployment to GitHub Container Registry.

## How It Works

The application operates as a simple web service with a single prediction endpoint.
1.  **API Server**: A `FastAPI` server provides the web interface. It's configured to run with `uvicorn`.
2.  **Model Loading**: On startup, the application downloads the pre-trained PyTorch model (`faces_best_model.pth`) from a pre-configured Amazon S3 bucket.
3.  **Preprocessing**: When an image is uploaded, it is resized to 224x224, converted to a PyTorch tensor, and normalized.
4.  **Inference**: The processed tensor is passed through the `efficientnet_b0` model.
5.  **Prediction**: The model's single logit output is passed through a sigmoid function to get the probability of the image being 'Real'. The final prediction (`is_fake`) and confidence score are calculated based on this probability.

## API Endpoint

### Predict Deepfake

Accepts an image file and returns a prediction on whether it is a deepfake.

-   **URL**: `/predict/`
-   **Method**: `POST`
-   **Body**: `multipart/form-data`
    -   `file`: The image file to be analyzed.

#### Example Success Response (200 OK)

```json
{
  "is_fake": true,
  "confidence": 0.987
}
```

-   `is_fake`: A boolean indicating if the image is classified as a deepfake.
-   `confidence`: The model's confidence in the `is_fake` prediction, calculated as `1.0 - probability_of_real`.

#### Example Error Response (400 Bad Request)

```json
{
    "message": "Error processing image: <error_details>"
}
```

## Getting Started

### Prerequisites

*   Docker

### Running Locally with Docker

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/saurav-kan/DeepfakeDetector.git
    cd DeepfakeDetector
    ```

2.  **Build the Docker image:**
    ```bash
    docker build -t deepfake-api .
    ```

3.  **Run the Docker container:**
    The application will be accessible at `http://localhost:8000`.
    ```bash
    docker run -p 8000:80 deepfake-api
    ```
    > **Note**: The default application loads the model from a specific S3 bucket (`deepfake-model-storage-saurav-2025`). If you wish to use your own model, you will need to modify the `load_model` function in `main.py`.

### Testing the API

You can use the provided `test_api.py` script or a tool like `curl`.

#### Using the Test Script

1.  Modify `test_api.py` and set `image_path` to the location of your test image.
    ```python
    # in test_api.py
    image_path = "path/to/your/image.jpg" # <-- IMPORTANT: CHANGE THIS PATH
    ```

2.  Run the script (ensure the container is running):
    ```bash
    pip install requests
    python test_api.py
    ```

#### Using `curl`

```bash
curl -X POST "http://127.0.0.1:8000/predict/" -F "file=@/path/to/your/image.jpg"
```

## Deployment

This repository is configured with a GitHub Actions workflow (`.github/workflows/deploy.yml`) for continuous integration and deployment.

-   **Trigger**: On every push to the `main` branch.
-   **Action**: The workflow builds the Docker image and pushes it to the GitHub Container Registry.
-   **Image Location**: `ghcr.io/saurav-kan/deepfake-api:latest`
-   **Secrets**: The workflow requires a `GHCR_TOKEN` repository secret with permissions to write packages to the registry.

## Project Structure

```
└── saurav-kan-deepfakedetector/
    ├── Dockerfile               # Defines the application's Docker container.
    ├── faces_best_model.pth     # The pre-trained PyTorch model weights.
    ├── main.py                  # The FastAPI application logic and API endpoints.
    ├── requirements.txt         # Python dependencies.
    ├── test_api.py              # A script to test the prediction endpoint.
    ├── verify.py                # A utility script to verify a PyTorch model file.
    └── .github/
        └── workflows/
            └── deploy.yml       # GitHub Actions workflow for CI/CD.
