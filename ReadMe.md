
# Sugarcane Plant Disease Detection System

This project is an end-to-end system for detecting sugarcane plant diseases using deep learning. The system consists of a Convolutional Neural Network (CNN) built from scratch using PyTorch for image classification, with a FastAPI backend to handle image uploads and provide predictions. The frontend, built with React.js, offers a user-friendly interface to upload images and display real-time results.

## Features
- **CNN-Based Classification:** Classifies sugarcane leaf health by analyzing uploaded images.
- **FastAPI Backend:** Handles image uploads and returns disease predictions.
- **React.js Frontend:** Simple interface for users to upload images and view results in real time.

## Demo
[Add demo link or screenshots if applicable]

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 14+
- PyTorch 1.9+
- FastAPI
- Uvicorn
- React.js

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MayukhTunga/SugarCane-Disease-Prediction.git
   cd SugarCane-Disease-Prediction
   ```

2. **Backend Setup (FastAPI):**
   - Install the required Python packages:
     ```bash
     pip install -r backend/requirements.txt
     ```
   - Start the FastAPI server:
     ```bash
     uvicorn backend.main:app --reload
     ```

3. **Frontend Setup (React.js):**
   - Install Node.js dependencies:
     ```bash
     cd frontend
     npm install
     ```
   - Start the React development server:
     ```bash
     npm start
     ```

## Usage
1. Navigate to the frontend at `http://localhost:3000`.
2. Upload a sugarcane leaf image.
3. Receive real-time classification results indicating the plant's health.

## Model Overview
The CNN model is trained to classify sugarcane diseases from images. It is implemented in PyTorch and fine-tuned on a custom dataset of sugarcane leaves showing various health conditions.

### Architecture
- **Input Layer:** Takes a sugarcane leaf image.
- **Hidden Layers:** Multiple convolutional, pooling, and fully connected layers.
- **Output Layer:** Predicts disease class from the image.

## API Documentation
The FastAPI backend exposes a simple API to upload images and get predictions.

- **POST /predict**
  - **Description:** Upload an image and receive a disease prediction.
  - **Parameters:** 
    - Image (JPEG/PNG)
  - **Response:** JSON object containing the prediction and confidence score.

Example:
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/image.jpg"
```

## Future Improvements
- Improve model accuracy by incorporating more data and training techniques.
- Add more advanced disease classification categories.
- Implement user authentication and image history tracking.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with any suggested improvements.

## License
[MIT License](LICENSE)

## Acknowledgements
- PyTorch for the deep learning framework.
- FastAPI for the backend framework.
- React.js for the frontend interface.

---
