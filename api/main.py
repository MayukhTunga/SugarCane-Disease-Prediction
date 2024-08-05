from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import torch
from torchvision.transforms import v2
print(torch.__version__)

app = FastAPI()

model = torch.load('Model.pth')
model.eval()
class_names = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']
mean = ([0.4977, 0.5281, 0.3800])
std = ([0.1744, 0.1733, 0.1814])

transforms = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.RandomVerticalFlip(),
    v2.RandomHorizontalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean, std) 
])

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
        predictions = torch.argmax(output, dim=1)
    pass


if __name__ == "__main__":
    uvicorn.run(app, host = 'localhost', port = 8080)

