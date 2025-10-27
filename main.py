import io
import boto3  # For S3
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # For Vercel
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

app = FastAPI(title="Deepfake Detector API")

# --- 1. ADD CORS FOR VERCEL ---
# This allows your Vercel frontend to call your API
origins = [
    "https://YOUR-VERCEL-APP-NAME.vercel.app",  # <-- IMPORTANT: Put your Vercel URL here
    "http://localhost:3000",                   # For local testing
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. MODEL DEFINITION ---
# This now matches your new lightweight, binary model
model = timm.create_model(
    "efficientnet_b0", 
    pretrained=False, 
    num_classes=1  # <-- KEY CHANGE: 1 output neuron
)

# --- 3. STARTUP EVENT (with S3) ---
@app.on_event("startup")
def load_model():
    """
    Load the new PyTorch model from S3 on startup.
    """
    try:
        # --- S3 Download ---
        BUCKET_NAME = "deepfake-model-storage-saurav-2025"  # <-- Your S3 bucket
        MODEL_KEY = "faces_best_model.pth"   # The file name in S3
        LOCAL_PATH = "/tmp/model.pth"        # Temp path
        
        print(f"Downloading model from S3: {BUCKET_NAME}/{MODEL_KEY}")
        s3 = boto3.client('s3')
        s3.download_file(BUCKET_NAME, MODEL_KEY, LOCAL_PATH)
        # --- End S3 ---

        # Load the new model
        model.load_state_dict(torch.load(LOCAL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print("Model loaded successfully.")
    
    except Exception as e:
        print(f"Error loading model: {e}")

# --- 4. PREPROCESSING (Unchanged) ---
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 5. PREDICT ENDPOINT (Updated) ---
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image, runs inference, and returns a binary prediction.
    """
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = preprocess_transform(image)
        input_batch = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_batch) # Output shape is [1, 1]

        # --- KEY CHANGE: Binary Logic ---
        # 1. Apply Sigmoid to the single logit
        # This gives the probability of being Class 1 ('Real')
        prob_real = torch.sigmoid(output).item()

        # 2. Probability of 'Fake' is the opposite
        prob_fake = 1.0 - prob_real

        # 3. Determine if it's fake
        is_fake = prob_fake >= 0.5
        # --- End Logic ---

        return {
            "is_fake": bool(is_fake),
            "confidence": float(prob_fake)
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"Error processing image: {e}"})