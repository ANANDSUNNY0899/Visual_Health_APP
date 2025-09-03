// src/pages/ReportsPage.jsx
// FINAL, COMPLETE VERSION WITH ALL FEATURES

import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';
import './ReportsPage.css';

// Sub-component to display the formatted AI summary
function ReportSummary({ summary }) {
  if (!summary) return null;

  // Helper function to format bullet points from the AI's markdown-like response
  const formatText = (text) => {
    return text.split('*').map((item, index) => 
      item.trim() ? <li key={index}>{item.trim()}</li> : null
    );
  };

  return (
    <div className="summary-container">
      <h3>AI-Powered Summary</h3>
      <div className="summary-section">
        <h4>Key Findings</h4>
        <ul>{formatText(summary.key_findings)}</ul>
      </div>
      <div className="summary-section">
        <h4>What It Means</h4>
        <p>{summary.what_it_means}</p>
      </div>
      <div className="summary-section">
        <h4>Next Steps to Discuss with Your Doctor</h4>
        <ul>{formatText(summary.next_steps)}</ul>
      </div>
    </div>
  );
}

// The main page component
export default function ReportsPage() {
  const [reports, setReports] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [summary, setSummary] = useState(null);

  // Fetches the list of past reports when the page first loads
  useEffect(() => {
    const fetchReports = async () => {
      try {
        const data = await api.get('/reports');
        setReports(data);
      } catch (error) {
        console.error("Failed to fetch reports:", error);
        setMessage(`Error: Could not load report history. ${error.message}`);
      }
    };
    fetchReports();
  }, []);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setMessage('');
    setSummary(null); // Clear any previous summary when a new file is chosen
  };

  const handleUploadAndSummarize = async () => {
    if (!selectedFile) {
      setMessage('Please select a PDF file first.');
      return;
    }
    setIsLoading(true);
    setMessage('Step 1/2: Uploading file...');
    setSummary(null);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Step 1: Upload the file. The backend returns the extracted text.
      const uploadData = await api.postForm('/reports/upload', formData);
      setReports(prev => [uploadData, ...prev]);
      
      // Step 2: Immediately call the summarize endpoint with the text we got back
      setMessage('Step 2/2: Analyzing report with AI...');
      const summaryData = await api.post('/reports/summarize', {
        report_text: uploadData.extracted_text
      });
      setSummary(summaryData);
      setMessage('Analysis complete!');

    } catch (error) {
      setMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
      setSelectedFile(null);
    }
  };

  // Handler for the "Clear History" button
  const handleClearHistory = async () => {
    const isConfirmed = window.confirm(
      "Are you sure you want to delete all your report history? This action cannot be undone."
    );

    if (isConfirmed) {
      setIsLoading(true);
      setMessage("Deleting history...");
      try {
        const response = await api.delete('/reports/clear');
        setMessage(response.message);
        setReports([]); // Immediately clear the reports from the UI
        setSummary(null); // Clear any visible summary
      } catch (error) {
        setMessage(`Error: ${error.message}`);
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="reports-page-container">
      <div className="reports-content">
        <Link to="/" className="back-link">← Back to Digital Twin</Link>
        <h2>Medical Reports</h2>
        
        <div className="upload-section">
          <h4>Upload New Report</h4>
          <p>Upload a PDF of your lab results to get an instant AI summary.</p>
          <div className="upload-box">
            <input type="file" id="file-upload" accept=".pdf" onChange={handleFileChange} disabled={isLoading} />
            <label htmlFor="file-upload" className="upload-label">{selectedFile ? selectedFile.name : 'Choose a PDF file...'}</label>
          </div>
          <button onClick={handleUploadAndSummarize} disabled={!selectedFile || isLoading}>
            {isLoading ? message : 'Upload & Analyze'}
          </button>
          {message && !isLoading && <p className={`message ${message.startsWith('Error') ? 'error' : 'success'}`}>{message}</p>}
        </div>
        
        {summary && <ReportSummary summary={summary} />}
        
        <div className="history-section">
          <div className="history-header">
            <h4>Report History</h4>
            {reports.length > 0 && !isLoading && (
              <button 
                onClick={handleClearHistory} 
                className="clear-history-btn" 
              >
                Clear History
              </button>
            )}
          </div>
          
          {reports.length > 0 ? (
            <ul className="reports-list">
              {reports.map(report => (
                <li key={report.id} className="report-item">
                  <span>{report.filename}</span>
                  <span>{new Date(report.uploaded_at).toLocaleDateString()}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="no-reports">You have not uploaded any reports yet.</p>
          )}
        </div>
      </div>
    </div>
  );
}