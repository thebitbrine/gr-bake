import base64
import io
import logging
import os
from typing import Dict, Any

import numpy as np
import torch
import torchvision.transforms as T
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForImageSegmentation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BiRefNet Image Segmentation API")

# Global model reference
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing transforms (same as BiRefNet inference scripts)
preprocess = T.Compose([
    T.Resize((1024, 1024)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model() -> None:
    global model
    logger.info("Loading BiRefNet model...")
    try:
        model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet",
            trust_remote_code=True
        )
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

@app.on_event("startup")
async def startup_event():
    load_model()

class InferRequest(BaseModel):
    image: str  # base64 encoded image

class InferResponse(BaseModel):
    mask: str  # base64 encoded PNG mask

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image)
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = image_pil.size  # (width, height)

        # Preprocess
        input_tensor = preprocess(image_pil).unsqueeze(0).to(device)  # shape (1,3,1024,1024)

        # Inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                output = model(input_tensor)

        # Handle different output formats
        if isinstance(output, torch.Tensor):
            logits = output
        elif hasattr(output, 'logits'):
            logits = output.logits
        elif isinstance(output, (list, tuple)) and len(output) > 0:
            logits = output[0]
        else:
            raise TypeError(f"Unexpected model output type: {type(output)}")

        logger.info(f"Model output shape: {logits.shape}")

        # Postprocess: sigmoid, threshold, resize to original, convert to PIL
        mask = torch.sigmoid(logits).squeeze().cpu().numpy()  # (1024,1024) float [0,1]
        mask_binary = (mask > 0.5).astype(np.uint8) * 255  # (1024,1024) uint8
        mask_pil = Image.fromarray(mask_binary).resize(original_size, Image.LANCZOS)

        # Encode to PNG base64
        buf = io.BytesIO()
        mask_pil.save(buf, format="PNG")
        buf.seek(0)
        mask_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return InferResponse(mask=mask_b64)

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
