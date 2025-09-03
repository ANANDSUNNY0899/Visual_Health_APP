VisuHealth: The Proactive AI Digital Twin
VisuHealth is not just a health tracker; it's a next-generation visual health prediction application that creates a personalized "Digital Twin" of the user. By combining user-provided lifestyle data, uploaded medical reports, and a powerful AI co-pilot, VisuHealth provides a unique, intuitive, and proactive way to understand the long-term impact of daily choices on your health.

 Key Features
VisuHealth is built on four powerful pillars, creating a comprehensive and intelligent health co-pilot.
 1. Dynamic 3D Health Simulation
Interactive 3D Model: A visual representation of internal organs that dynamically changes based on user inputs.
Time Projection: A powerful time-lapse slider simulates the cumulative effects of habits over decades, showing potential future damage or healing.
Real-time Visual Feedback: Organs change color and animate with a "pulsing" glow to indicate stress levels, providing immediate, intuitive feedback.

 2. AI-Powered Conversational Assistant (RAG)
Chat with Your Twin: A full-featured chat interface allows users to ask complex health questions in natural language.
Report-Aware Context: The AI uses the content of the user's uploaded medical reports as a primary source of truth, providing hyper-personalized answers.
Multi-Language Support: The AI can generate responses in multiple languages (currently English and Hindi) to improve accessibility.
Persistent Chat History: The application saves all conversations, allowing users to revisit past discussions and maintain context with their AI assistant.

 3. Deep Personalization & Data Management
Secure User Authentication: Full user registration, login, and logout system using JWT tokens to protect all user data.
AI Medical Report Analysis: Users can upload PDF lab reports. The AI automatically extracts the text, generates a structured summary (Key Findings, What It Means, Next Steps), and displays it on a dedicated dashboard.
Biometric Logging & Charting: A "My Health Log" page where users can manually track key metrics like weight and blood pressure over time, with their trends visualized on an interactive chart.

 4. Proactive Health Insights & Alerts
Autonomous Backend Scheduler: A background process runs automatically to analyze the user's latest reports and data.
AI-Generated Notifications: If the AI finds a noteworthy risk or trend, it generates a proactive health insight.
Notification Center: A polished UI with a bell icon in the header displays unread insights, helping users stay informed about their health without having to ask.

5.System Architecture
VisuHealth is a full-stack application composed of a React frontend, a FastAPI backend, and a connection to the Google Gemini API.
code
Mermaid
graph TD
    A[User's Browser] -- Interacts --> B(React Frontend);
    
    B -- Sends API Requests (REST) --> C{FastAPI Backend};
    
    C -- Reads/Writes Data --> D[(SQLite Database)];
    
    C -- Augments Prompt --> E(Google Gemini API);
    E -- Generates Text --> C;
    
    F(Background Scheduler) -- Triggers Analysis --> C;

    subgraph "Frontend (Vite)"
        B
    end

    subgraph "Backend (Python)"
        C
        D
        F
    end
6.Technology Stack
Category	Technology / Library
Frontend	React, Vite, React Router, Recharts, Three.js (@react-three/fiber, @react-three/drei)
Backend	Python, FastAPI, Uvicorn, SQLAlchemy
AI / ML	Google Gemini API, Sentence Transformers, Faiss, PyPDF2
Database	SQLite (for development)
Scheduling	APScheduler
Authentication	Passlib (bcrypt), python-jose (JWT)


7.Getting Started
Follow these instructions to set up and run the project locally.
Prerequisites
Node.js and npm (for the frontend)
Python 3.10+ and pip (for the backend)
A Google Gemini API Key



1. Backend Setup
  Clone the repository:
   git clone <your-repo-url>
   cd visuhealth-app
  2.Create the Environment File: In the root directory (visuhealth-app/), create a file named .env and add your Google Gemini API key:
  GOOGLE_API_KEY="your_api_key_here"

  3.Install Python Dependencies:
  pip install -r requirements.txt 
  # Or install manually: fastapi uvicorn sqlalchemy pypdf2 "passlib[bcrypt]" python-jose google-generativeai sentence-transformers faiss-cpu pydantic-settings python-dotenv apscheduler

  4. Delete Old Database (Important for first run): If a visuhealth.db file exists in the backend/ folder, delete it to ensure the latest database schema is created.


2. Frontend Setup
   Install Node Modules: In a new terminal, from the root directory, run:

   npm install

   3. Running the Full Application
     VisuHealth requires two servers to run simultaneously.
     Terminal 1: Start the Backend
     From the root visuhealth-app directory, run:
     python run.py

    Terminal 2: Start the Frontend
    From the root visuhealth-app directory, run:
    npm run dev


Your browser should open to http://localhost:5173, or you can navigate there manually.

You can now register a new user and explore all the features of the application.
