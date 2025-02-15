import { useState } from "react";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import RoadmapModal from "./components/Modal";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [roadmapData, setRoadmapData] = useState(null);

  const analyzePitch = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('http://localhost:8000/api/pitch/analyze', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Analysis failed: ${response.statusText}`);
    }

    return response.json();
  };

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      setError("Please select a PDF file");
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const result = await analyzePitch(selectedFile);
      setAnalysis(result);
      setShowSuggestions(true);
    } catch (err) {
      setError("Failed to analyze pitch deck: " + err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleRoadmapSubmit = async ({ startupName, timeline }) => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/api/roadmap/create', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: 1,
          startup_name: startupName,
          timeline: parseInt(timeline)
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to generate roadmap');
      }
      
      const data = await response.json();
      setRoadmapData(data.roadmap);
      setIsModalOpen(false);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <Navbar />
      <main>
        <div className="file-upload-container">
          <input type="file" accept=".pdf" onChange={handleFileChange} />
          <button onClick={handleSubmit} disabled={loading}>
            {loading ? "Processing..." : "Submit"}
          </button>
          <button onClick={() => setIsModalOpen(true)}>
            Generate Roadmap
          </button>
        </div>

        <RoadmapModal
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          onSubmit={handleRoadmapSubmit}
        />

        {error && <div className="error">{error}</div>}

        {roadmapData && (
          <div className="roadmap-display">
            <h3>{roadmapData.startup_name} - {roadmapData.timeline} Month Roadmap</h3>
            <pre>{roadmapData.roadmap_text}</pre>
          </div>
        )}

        {analysis && (
          <div>
            <h2>Analysis Output</h2>
            <div>
              <h3>Gemini Analysis</h3>
              <pre>{analysis.gemini_analysis}</pre>
            </div>
            
            <h2>Development Roadmap</h2>
            <button onClick={() => setShowSuggestions(!showSuggestions)}>
              {showSuggestions ? 'Hide' : 'Show'} Suggestions
            </button>
            
            {showSuggestions && analysis.ml_prediction && (
              <div>
                <h3>ML Predictions</h3>
                <p>Acceptance Probability: {(analysis.ml_prediction.probability * 100).toFixed(2)}%</p>
                <h4>Suggestions:</h4>
                <ul>
                  {analysis.ml_prediction.suggestions.map((suggestion, index) => (
                    <li key={index}>{suggestion}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </main>
      <Footer />
    </div>
  );
}

export default App;
