from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import torch 
import torch.nn as nn
from torchvision.transforms import v2
import torch.nn.functional as F
import io

app = FastAPI()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*53*53,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = Net()

mean = ([0.4977, 0.5281, 0.3800])
std = ([0.1744, 0.1733, 0.1814])

transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean, std) 
])


PATH = 'Model_State_Dict.pth'
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()
class_names = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

@app.get("/ping")
async def ping():
    return "Hello the server is running"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = transform(image).unsqueeze(0)  

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        predicted_class_idx = predicted.item()
        predicted_class_name = class_names[predicted_class_idx]

    return JSONResponse(content={"predicted_class": predicted_class_name})


if __name__ == "__main__":
    uvicorn.run(app, host = 'localhost', port = 8080)

