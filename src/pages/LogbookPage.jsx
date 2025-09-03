// src/pages/LogbookPage.jsx
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';
import './LogbookPage.css'; // We will create this
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// A map to make our metric names user-friendly
const METRIC_CONFIG = {
  weight_kg: { name: 'Body Weight', unit: 'kg' },
  blood_pressure_systolic: { name: 'Blood Pressure (Systolic)', unit: 'mmHg' },
  blood_pressure_diastolic: { name: 'Blood Pressure (Diastolic)', unit: 'mmHg' },
  blood_glucose: { name: 'Blood Glucose', unit: 'mg/dL' },
};

export default function LogbookPage() {
  const [activeMetric, setActiveMetric] = useState('weight_kg');
  const [history, setHistory] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState('');

  // Fetch data whenever the active metric changes
  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        const data = await api.get(`/logbook/${activeMetric}`);
        setHistory(data.map(item => ({
          ...item,
          // Format the date for the chart
          logged_at_formatted: new Date(item.logged_at).toLocaleDateString(),
        })));
      } catch (error) {
        console.error(`Failed to fetch ${activeMetric} data:`, error);
        setMessage(`Error: Could not load data for ${METRIC_CONFIG[activeMetric].name}.`);
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, [activeMetric]);

  const handleLogSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue) return;

    try {
      const newData = await api.post('/logbook', {
        metric_type: activeMetric,
        value: parseFloat(inputValue),
      });
      // Add the new data to our history to instantly update the UI
      setHistory(prev => [...prev, {
        ...newData,
        logged_at_formatted: new Date(newData.logged_at).toLocaleDateString(),
      }]);
      setInputValue('');
      setMessage('Log saved successfully!');
    } catch (error) {
      console.error('Failed to log data:', error);
      setMessage(`Error: ${error.message}`);
    }
  };

  return (
    <div className="logbook-page-container">
      <div className="logbook-content">
        <Link to="/" className="back-link">← Back to Digital Twin</Link>
        <h2>My Health Log</h2>
        
        <div className="metric-selector">
          {Object.entries(METRIC_CONFIG).map(([key, { name }]) => (
            <button
              key={key}
              className={`metric-btn ${activeMetric === key ? 'active' : ''}`}
              onClick={() => setActiveMetric(key)}
            >
              {name}
            </button>
          ))}
        </div>

        <div className="logbook-main">
          <div className="log-form-section">
            <h4>Log New Entry</h4>
            <form onSubmit={handleLogSubmit}>
              <label htmlFor="metric-value">{METRIC_CONFIG[activeMetric].name} ({METRIC_CONFIG[activeMetric].unit})</label>
              <input
                id="metric-value"
                type="number"
                step="0.1"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Enter value..."
                required
              />
              <button type="submit">Save Log</button>
              {message && <p className="log-message">{message}</p>}
            </form>
          </div>

          <div className="chart-section">
            <h4>Your Trend</h4>
            {history.length > 1 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={history}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis dataKey="logged_at_formatted" stroke="#aaa" />
                  <YAxis stroke="#aaa" domain={['dataMin - 5', 'dataMax + 5']} />
                  <Tooltip contentStyle={{ backgroundColor: '#2a2a2a', border: '1px solid #444' }} />
                  <Legend />
                  <Line type="monotone" dataKey="value" name={METRIC_CONFIG[activeMetric].name} stroke="var(--accent-color)" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <p className="no-data-message">Log at least two entries to see a trend chart.</p>
            )}
          </div>
        </div>

        <div className="history-section">
          <h4>Log History</h4>
          <div className="history-table">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Value ({METRIC_CONFIG[activeMetric].unit})</th>
                </tr>
              </thead>
              <tbody>
                {history.slice().reverse().map(log => (
                  <tr key={log.id}>
                    <td>{new Date(log.logged_at).toLocaleString()}</td>
                    <td>{log.value.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {history.length === 0 && !isLoading && <p className="no-data-message">No logs found for this metric.</p>}
          </div>
        </div>
      </div>
    </div>
  );
}