import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setError('Please select an image first.');
      return;
    }
    setIsLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const apiUrl = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';
      const response = await fetch(`${apiUrl}/predict/`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Failed to analyze image. Is the server running?');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="App">
      <Navbar />
      <main className="main-content">
        <div className="detector-container">
          <h1>Deepfake Image Detector</h1>
          {!result && (
            <div className="upload-section">
              <form onSubmit={handleSubmit}>
                <label htmlFor="file-upload" className="custom-file-upload">
                  {preview ? <img src={preview} alt="Preview" /> : 'Click to select an image'}
                </label>
                <input id="file-upload" type="file" onChange={handleFileChange} accept="image/*" />
                {selectedFile && <button type="submit" disabled={isLoading}>{isLoading ? 'Analyzing...' : 'Analyze Image'}</button>}
              </form>
              {error && <p className="error-message">{error}</p>}
            </div>
          )}
          {isLoading && <div className="loader"></div>}
          {result && (
            <div className="result-section">
              <h2>Analysis Result</h2>
              <div className={result.is_fake ? 'result-card fake' : 'result-card real'}>
                <p className="result-label">{result.is_fake ? 'Deepfake Detected' : 'Authentic Image'}</p>
                <p className="confidence-score">Confidence: {Math.round(result.confidence * 100)}%</p>
              </div>
              <button onClick={handleReset}>Analyze Another Image</button>
            </div>
          )}
        </div>
        <section className="info-section">
          <h2>How It Works</h2>
          <p>
            This tool leverages a sophisticated deep learning model (EfficientNet-B4) trained on a vast dataset of real and synthetically generated images. When you upload an image, it's processed and analyzed for subtle artifacts and inconsistencies that are often invisible to the human eye but are tell-tale signs of digital manipulation. Our API provides a confidence score, giving you a clear indication of the likelihood of an image being a deepfake.
          </p>
        </section>
      </main>
      <Footer />
    </div>
  );
}

export default App;
