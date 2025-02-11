// frontend/src/App.js
import React, { useState } from "react";
import "./App.css";
import Visualizations from "./Visualizations";

// Helper function to submit the query using fetch
const submitQuery = async (query) => {
  try {
    const response = await fetch("http://localhost:8000/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ query })
    });
    
    // Debug: log the raw response status
    console.log("Response status:", response.status);
    
    // If the response is not OK, throw an error
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log("Response JSON:", data);
    return data;
  } catch (error) {
    console.error("Error retrieving data:", error);
    throw error;
  }
};

function App() {
  // State variables for the query result, error and loading indicator
  const [queryData, setQueryData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  // Handle form submission by calling submitQuery and updating state accordingly
  const handleSubmit = async (query) => {
    setError(null);       // Clear any previous error
    setQueryData(null);   // Clear previous data
    setLoading(true);
    try {
      const data = await submitQuery(query);
      console.log("Received data:", data);
      // Check that the data is in the expected format
      if (
        data &&
        data.answer &&
        data.sources &&
        data.sentiment &&
        data.topics
      ) {
        setQueryData(data);
      } else {
        console.warn("Response format not as expected:", data);
        setError("Error retrieving data. Please try again.");
      }
    } catch (err) {
      setError("Error retrieving data. Please try again.");
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <h1>Market Research RAG System</h1>
      {/* Simple form to submit the query */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          const query = e.target.elements.query.value;
          handleSubmit(query);
        }}
      >
        <input type="text" name="query" placeholder="Enter your query" />
        <button type="submit">Submit</button>
      </form>

      {/* Display errors if any */}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {/* Show a loading message if the API call is in progress */}
      {loading && <p>Loading...</p>}

      {/* Render the response only when valid data is received */}
      {queryData ? (
        <div>
          <h2>Answer:</h2>
          <p>{queryData.answer || "No answer available"}</p>
          <h3>Sources:</h3>
          <ul>
            {queryData.sources?.map((src, index) => (
              <li key={index}>{src}</li>
            ))}
          </ul>
          <h3>Sentiment:</h3>
          <p>{queryData.sentiment || "Unknown"}</p>
          <h3>Topics:</h3>
          <ul>
            {(queryData.topics || []).map((topic, index) => (
              <li key={index}>{topic}</li>
            ))}
          </ul>
          {/* Visualizations component for additional insights */}
          <Visualizations sentiment={queryData.sentiment} topics={queryData.topics} />
        </div>
      ) : (
        // If there's no error or loading state, show a placeholder
        !error && !loading && <p>No data yet.</p>
      )}
    </div>
  );
}

export default App;

