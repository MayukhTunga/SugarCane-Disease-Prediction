from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import torch 
import torch.nn as nn
from torchvision.transforms import v2
import torch.nn.functional as F

app = FastAPI()

mean = ([0.4977, 0.5281, 0.3800])
std = ([0.1744, 0.1733, 0.1814])

transforms = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.Resize((224, 224)),  
    v2.RandomVerticalFlip(),
    v2.RandomHorizontalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean, std) 
])

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

PATH = './Model.pth'
model.load_state_dict(torch.load(PATH))
model.eval()
class_names = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

@app.get("/ping")
async def ping():
    return "Hello the server is running"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    input_tensor = transforms(image)
    input_tensor = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.max(output.data, 1)
    pass


if __name__ == "__main__":
    uvicorn.run(app, host = 'localhost', port = 8080)

