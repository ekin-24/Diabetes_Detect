import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from torchvision import models, transforms
from PIL import Image

app = FastAPI(title="Diabetes Detect From Eye")

# Html path
app.mount("/static", StaticFiles(directory="static"), name="static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resnet50 model setup
model = models.resnet50(weights=None)
num_fc = model.fc.in_features
model.fc = nn.Linear(num_fc, 5)
model.load_state_dict(torch.load("retina_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

CLASS_NAMES = [
    "No Diagnosis (Healthy)", 
    "Stage 1: Mild Damage", 
    "Stage 2: Moderate Damage", 
    "Stage 3: Severe Damage", 
    "Stage 4: Proliferative"
]

@app.post("/predict/")
async def predict_retina(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            
            # Convert outputs to softmax
            probabilities = F.softmax(outputs, dim=1)[0] * 100
            
            # Find most accuracy rate and class
            max_prob, predicted_class = torch.max(probabilities, 0)
            
        result_idx = predicted_class.item()
        confidence = round(max_prob.item(), 2)
        
        return {
            "status": "success",
            "diagnosis": CLASS_NAMES[result_idx],
            "confidence": confidence
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

# First get index.html
@app.get("/")
def read_root():
    return FileResponse("static/index.html")