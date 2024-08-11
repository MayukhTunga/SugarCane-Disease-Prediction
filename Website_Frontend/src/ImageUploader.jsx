import React, { useState } from "react";
import axios from "axios";

function SugarCaneDiseaseIdentifier() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [result, setResult] = useState("");
    const [error, setError] = useState(null); 

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);
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
        <div className="flex flex-col items-center justify-center h-auto text-center mx-auto mt-4 mb-56 isolate aspect-video w-96 rounded-xl bg-white/20 shadow-lg ring-1 ring-black/5">
            <input 
                type="file" 
                className="m-4 p-4 border-2 rounded-lg cursor-pointer bg-white/50"
                onChange={handleFileChange} 
            />
            {selectedFile && (
                <div className="my-4">
                    <img 
                        src={URL.createObjectURL(selectedFile)} 
                        alt="Selected file" 
                        className="rounded-lg border-2 border-gray-300 w-[224px] h-[224px] object-cover"
                    />
                </div>
            )}
            <button 
                onClick={handleAnalyzeImage} 
                className="py-2 px-4 bg-green-600 text-white rounded-lg cursor-pointer text-lg"
            >
                Analyze Image
            </button>
            <div className="mt-6">
                <div className="flex items-center mb-4">
                    <h3 className="text-white text-3xl mr-1">Result:</h3>
                    {result && <p className="text-lime-300 text-3xl">{result}</p>}
                    {error && <p className="text-red-500 text-3xl">{error}</p>}
                </div>
            </div>
        </div>
    );
}

export default SugarCaneDiseaseIdentifier;
