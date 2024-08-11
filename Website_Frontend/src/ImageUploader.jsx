import React, { useState } from "react";
import axios from "axios";

function SugarCaneDiseaseIdentifier() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [result, setResult] = useState("");
    const [error, setError] = useState(null); 

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
        setResult(""); 
        setError(null); 
    };

    const handleAnalyzeImage = async () => {
        if (!selectedFile) {
            alert("Please select a file first!");
            return;
        }

        const formData = new FormData();
        formData.append("file", selectedFile); 

        const apiEndpoint = "http://localhost:8080/predict";

        try {
            const response = await axios.post(apiEndpoint, formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });

            setResult(response.data.predicted_class);
        } catch (error) {
            setError("An error occurred while processing the image.");
            console.error("There was an error!", error);
        }
    };

    return (
        <div className="flex-col items-center justify-center h-64 text-center mx-auto mt-4 mb-56 isolate aspect-video w-96 rounded-xl bg-white/20 shadow-lg ring-1 ring-black/5">
            <input 
                type="file" 
                className="m-4 p-4 border-2 rounded-lg cursor-pointer bg-white/50"
                onChange={handleFileChange} 

            />
            <button 
                onClick={handleAnalyzeImage} 
                style={styles.button}
            >
                Analyze Image
            </button>
            <div style={styles.resultContainer}>
                <div className="flex justify-center">
                    <h3 className="text-white text-3xl">Result:</h3>
                    {result && <p className="text-lime-300 text-3xl"> {result}</p>}
                    {error && <p className="text-white text-3xl">{error}</p>}
                </div>
            </div>
        </div>
    );
}

const styles = {
    container: {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100vh",
        backgroundColor: "#f4f4f4",
        textAlign: "center",
    },
    title: {
        fontSize: "2.5rem",
        color: "#333",
        marginBottom: "20px",
    },
    fileInput: {
        marginBottom: "20px",
        padding: "10px",
        borderRadius: "5px",
        border: "1px solid #ccc",
        cursor: "pointer",
    },
    button: {
        padding: "10px 20px",
        backgroundColor: "#4CAF50",
        color: "white",
        border: "none",
        borderRadius: "5px",
        cursor: "pointer",
        fontSize: "1rem",
    },
    resultContainer: {
        marginTop: "20px",
    },
    resultTitle: {
        fontSize: "1.5rem",
        color: "#333",
    },
    result: {
        fontSize: "1.2rem",
        color: "#4CAF50",
    },
    error: {
        fontSize: "1.2rem",
        color: "red",
    },
};

export default SugarCaneDiseaseIdentifier;
