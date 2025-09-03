// src/App.jsx
import React, { useState } from 'react';
import { Routes, Route, useNavigate, Navigate } from 'react-router-dom';
import LoginPage from './pages/LoginPage.jsx';
import RegisterPage from './pages/RegisterPage.jsx';
import HealthVisualizerApp from './HealthVisualizerApp.jsx';
import ReportsPage from './pages/ReportsPage.jsx';
import LogbookPage from './pages/LogbookPage.jsx';

export default function App() {
  const [token, setToken] = useState(localStorage.getItem('visuhealth_token'));
  // ... (rest of the component is the same)
  const navigate = useNavigate();

  const handleLoginSuccess = (newToken) => {
    localStorage.setItem('visuhealth_token', newToken);
    setToken(newToken);
  };

  const handleLogout = () => {
    localStorage.removeItem('visuhealth_token');
    setToken(null);
    navigate('/login');
  };

  return (
    <Routes>
      <Route path="/login" element={<LoginPage onLoginSuccess={handleLoginSuccess} />} />
      <Route path="/register" element={<RegisterPage />} />
      <Route
        path="/"
        element={
          token ? <HealthVisualizerApp onLogout={handleLogout} /> : <Navigate to="/login" />
        }
      />
      {/* 2. ADD THE NEW ROUTE for our reports page */}
      <Route 
        path="/reports"
        element={
          token ? <ReportsPage /> : <Navigate to="/login" />
        }
      />

      <Route 
      path="/reports"
      element={ token ? <ReportsPage /> : <Navigate to="/login" /> }
    />
    {/* 3. ADD THE NEW ROUTE */}
    <Route 
      path="/logbook"
      element={ token ? <LogbookPage /> : <Navigate to="/login" /> }
    />
    </Routes>
  );
}