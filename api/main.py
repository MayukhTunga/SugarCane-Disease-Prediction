from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np

app = FastAPI()


@app.get("/ping")
async def ping():
    return "Hello the server is running"

def read_file_as_image(data) -> np.ndarray:

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return


if __name__ == "__main__":
    uvicorn.run(app, host = 'localhost', port = 8080)