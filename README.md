# Market Research Retrieval-Augmented Generation (RAG) System

This project is a web application that lets users explore and compare market research reports using an AI-powered Retrieval-Augmented Generation (RAG) pipeline.

## Features

- **RAG Pipeline:** Retrieves relevant document segments from market research reports using semantic search (via FAISS) and generates insights with a language model.
- **Language Model Integration:** Uses Hugging Face’s text-generation pipeline (e.g., *gpt2*) to generate answers.
- **Additional AI Features:** Provides sentiment analysis and simple topic extraction from the retrieved context.
- **Interactive Frontend:** Built with React; supports responsive design, loading states, error handling, and detailed source views via double-click.
- **Testing:** Backend tests are provided using pytest. (Frontend tests can be added using Jest/React Testing Library.)

## Technologies Used

- **Backend:** Python, FastAPI, SentenceTransformers, FAISS, Hugging Face Transformers, PyTorch.
- **Frontend:** React, Axios.
- **Testing:** pytest for backend.

## File Structure

project/
├──── api/ 
│     └── backend/
│         ├── main.py
│         ├── requirements.txt
│         └── tests/
│               └── test_main.py
├── data/
│   └── reports 
│       
│       
├── frontend/
│   ├── package.json
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── App.js
│       ├── App.css
│       └── index.js
└── README.md


## Setup Instructions

### Backend

1. **Prepare the Data:**  
   Download the market research reports from the provided Google Drive link and place them as `.pdf` files in the `data/reports` folder.

2. **Install Dependencies:**  
   Navigate to the `backend` folder and install the required packages:
   ```bash
   pip install -r requirements.txt

3. **Run the Server:**
   Start the FastAPI backend with:
   ```bash
   uvicorn main:app --reload
   The API will be available at http://localhost:8000.

4. **Run Tests:**
   first check installation:
   Run  pip show pytest
   if not installed, install it with: 
   pip install pytest	
   Execute backend tests using:
   ```bash
   pytest tests/

### Frontend

1. **Install Dependencies:**
   Navigate to the frontend folder and install the Node packages:
   ```bash
   npm install

2. **Start the Development Server:**
   Run the React app with:
   ```bash
   npm start
   The application will open at http://localhost:3000.

### Deployment
   **Backend: Deploy on platforms such as Heroku, Render, or any server supporting FastAPI.**
   **Frontend: Deploy on services like Vercel, Netlify, or similar.**
   **CORS: Ensure CORS settings are correctly configured for production.**

## Future Enhancements
   Replace the naive topic extraction with a more advanced topic modeling solution.
   Integrate a more robust language model for higher-quality AI-generated insights.
   Add caching to improve performance.
   Enhance the frontend with additional data visualizations and UX improvements.
   Implement user authentication and persistent query histories.


### Example of Queries to experiment with the app
 Here are some recommended natural language queries that users might submit through the frontend to explore and compare the market research reports:

 **Emerging Trends:**
 "What are the emerging market trends mentioned in the reports?"

 **Comparative Analysis:**
 "How do the consumer insights in Report 1 differ from those in Report 2?"

 **Competitive Landscape:**
 "Summarize the competitive analysis provided in the reports."

 **Market Forecasts:**
 "What market forecasts are discussed in the reports, and how do they compare?"

 **Sentiment Overview:**
 "What overall sentiment do the reports convey about the current market situation?"

 **Key Topics Extraction:**
 "Identify the key topics or themes mentioned across the reports."

 **Actionable Insights:**
 "What actionable insights or recommendations can be derived from these reports?"

 **Risk and Opportunity Analysis:**
 "What risks and opportunities are highlighted in the reports?"

 PS: Feel free to experiment with these queries or modify them to fit your specific interests when exploring the market research data.   

### License
This project is licensed under the MIT License.

---

## Final Notes

- **Customization:** Replace the dummy or simple functions (like topic extraction) with production‑grade methods if desired.
- **Model Upgrades:** You can swap out `gpt2` for a more advanced model from Hugging Face (or use OpenAI’s API) as needed.
- **Frontend Enhancements:** Feel free to extend the React code for better state management (e.g., using Redux), more advanced error handling, or richer visualizations.
- **Testing:** Additional tests (frontend unit/integration tests) can be added to further ensure robustness.

This complete solution should serve as a strong foundation that demonstrates best practices in backend architecture, RAG integration   
