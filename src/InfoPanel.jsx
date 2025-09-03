// src/InfoPanel.jsx
import React from 'react';
import './InfoPanel.css';

// --- MODIFIED: Added Heart data ---
const organData = {
  Lungs: {
    function: "The lungs are essential for respiration. They bring oxygen into the bloodstream and remove carbon dioxide.",
    risks: "Smoking introduces tar and thousands of chemicals, leading to inflammation, reduced capacity, and increased risk of Chronic Obstructive Pulmonary Disease (COPD) and cancer."
  },
  Liver: {
    function: "The liver is a vital organ that filters toxins from the blood, aids in digestion, and produces essential proteins.",
    risks: "Excessive alcohol consumption can lead to inflammation (alcoholic hepatitis), scarring (cirrhosis), and fatty liver disease, severely impairing its ability to function."
  },
  Heart: {
    function: "The heart is a muscular organ that pumps blood through the circulatory system, supplying oxygen and nutrients to the body.",
    risks: "A poor diet high in saturated fats and sugars can lead to atherosclerosis (plaque buildup in arteries), increasing the risk of heart attack, stroke, and high blood pressure."
  }
};

export default function InfoPanel({ organName, stressLevel, onClose, prediction }) {
  // The rest of this file can remain exactly the same, no changes needed below this line.
  const data = organData[organName];

  const getRiskLevelText = () => {
    if (stressLevel < 0.3) return "Low Risk";
    if (stressLevel < 0.7) return "Moderate Risk";
    return "High Risk";
  };

  return (
    <div className="info-panel">
      <div className="info-header">
        <h2>{organName}</h2>
        <button className="close-btn" onClick={onClose}>X</button>
      </div>
      <div className="info-content">
        <h3>Function:</h3>
        <p>{data.function}</p>
        
        <h3>Risks from Your Habits:</h3>
        <p>{data.risks}</p>
        
        <div className="risk-level">
          <strong>Visual Risk Level:</strong> {getRiskLevelText()}
        </div>

        {prediction && prediction.disease && (
          <div className="prediction-section">
            <h3>AI Disease Risk Prediction:</h3>
            <p className="prediction-score">
              <span>{prediction.disease}:</span>
              <strong>{prediction.riskScore}%</strong>
            </p>
          </div>
        )}
      </div>
    </div>
  );
}