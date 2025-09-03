// // src/HealthVisualizerApp.jsx
// // This is the final, complete version with all features integrated.

// import React, { useState, useMemo, useEffect } from 'react';
// import HealthVisualizer from './HealthVisualizer.jsx';
// import InfoPanel from './InfoPanel.jsx';
// import Header from './Header.jsx';
// import ChatPanel from './ChatPanel.jsx'; // 1. Import the ChatPanel
// import './App.css';

// const MAX_YEARS = 30;

// export default function HealthVisualizerApp({ onLogout }) {
//   // --- All state variables, including 'user' ---
//   const [user, setUser] = useState(null);
//   const [lifestyle, setLifestyle] = useState({ Lungs: 0, Liver: 0, Heart: 0 });
//   const [timeProjection, setTimeProjection] = useState(0);
//   const [selectedOrgan, setSelectedOrgan] = useState(null);
//   const [prediction, setPrediction] = useState(null);
//   const [isChatVisible, setIsChatVisible] = useState(false);

//   // --- 2. useEffect to fetch user data on load ---
//   useEffect(() => {
//     const fetchUserData = async () => {
//       const token = localStorage.getItem('visuhealth_token');
//       if (!token) {
//         onLogout();
//         return;
//       }

//       try {
//         const response = await fetch('http://127.0.0.1:8000/users/me/', {
//           headers: {
//             'Authorization': `Bearer ${token}`
//           }
//         });
//         if (response.ok) {
//           const userData = await response.json();
//           setUser(userData);
//         } else {
//           onLogout();
//         }
//       } catch (error) {
//         console.error("Failed to fetch user data:", error);
//         onLogout(); // Log out on network errors too
//       }
//     };
//     fetchUserData();
//   }, []); // Empty array ensures this runs only once when the component mounts

//   // --- All handler functions and memoized calculations ---
//   const handleLifestyleChange = (organ, value) => {
//     setLifestyle(prevState => ({ ...prevState, [organ]: parseFloat(value) }));
//   };

//   const projectedStress = useMemo(() => {
//     const timeFactor = timeProjection / MAX_YEARS;
//     return {
//       Lungs: lifestyle.Lungs * timeFactor,
//       Liver: lifestyle.Liver * timeFactor,
//       Heart: lifestyle.Heart * timeFactor,
//     };
//   }, [lifestyle, timeProjection]);

//   const handleOrganClick = async (organName) => {
//     setSelectedOrgan(organName);
//     setPrediction(null);
//     const token = localStorage.getItem('visuhealth_token');
//     if (!token) { onLogout(); return; }

//     try {
//       let endpoint = '';
//       let payload = {};

//       if (organName === 'Liver') {
//         endpoint = 'http://127.0.0.1:8000/predict/liver';
//         payload = { alcoholIntensity: lifestyle.Liver, timeProjection };
//       } else if (organName === 'Heart') {
//         endpoint = 'http://127.0.0.1:8000/predict/heart';
//         payload = { dietIntensity: lifestyle.Heart, smokingIntensity: lifestyle.Lungs, timeProjection };
//       }

//       if (endpoint) {
//         const response = await fetch(endpoint, {
//           method: 'POST',
//           headers: {
//             'Content-Type': 'application/json',
//             'Authorization': `Bearer ${token}`
//           },
//           body: JSON.stringify(payload),
//         });
//         const data = await response.json();
//         setPrediction(data);
//       }
//     } catch (error) {
//       console.error("Error fetching prediction:", error);
//       setPrediction({ disease: 'Prediction Server Offline', riskScore: 'N/A' });
//     }
//   };

//   const handleClosePanel = () => {
//     setSelectedOrgan(null);
//     setPrediction(null);
//   };

//   // --- 3. Data structure needed for the ChatPanel ---
//   const lifestyleDataForChat = {
//     smokingIntensity: lifestyle.Lungs,
//     alcoholIntensity: lifestyle.Liver,
//     dietIntensity: lifestyle.Heart,
//     timeProjection: timeProjection
//   };
// return (
//     <div className="app-container">
//       <Header userEmail={user ? user.email : 'Loading...'} onLogout={onLogout} />
      
//       <main className="main-content">
//         {/* --- UI Elements --- */}
//         <div className="ui-panel">
//           <h1>Health Controls</h1>
//           <div className="slider-group"><label>Smoking Intensity (Lungs)</label><input type="range" min="0" max="1" step="0.01" value={lifestyle.Lungs} onChange={(e) => handleLifestyleChange('Lungs', e.target.value)} /></div>
//           <div className="slider-group"><label>Alcohol Intensity (Liver)</label><input type="range" min="0" max="1" step="0.01" value={lifestyle.Liver} onChange={(e) => handleLifestyleChange('Liver', e.target.value)} /></div>
//           <div className="slider-group"><label>Poor Diet Intensity (Heart)</label><input type="range" min="0" max="1" step="0.01" value={lifestyle.Heart} onChange={(e) => handleLifestyleChange('Heart', e.target.value)} /></div>
//           <div className="slider-group time-slider"><label>Time Projection: {timeProjection} Years</label><input type="range" min="0" max={MAX_YEARS} step="1" value={timeProjection} onChange={(e) => setTimeProjection(parseInt(e.target.value))} /></div>
//         </div>
        
//         <button className="chat-fab" onClick={() => setIsChatVisible(true)}>
//           Chat with Twin
//         </button>

//         {selectedOrgan && (
//           <InfoPanel 
//             organName={selectedOrgan}
//             stressLevel={projectedStress[selectedOrgan]}
//             onClose={handleClosePanel}
//             prediction={prediction}
//           />
//         )}
        
//         <ChatPanel
//           lifestyleData={lifestyleDataForChat}
//           isVisible={isChatVisible}
//           onClose={() => setIsChatVisible(false)}
//         />
        
//         {/* --- 3D Canvas in its own container, at the back --- */}
//         <div className="canvas-container">
//           <HealthVisualizer organStress={projectedStress} onOrganClick={handleOrganClick} />
//         </div>
//       </main>
//     </div>
//   );
// }





// src/HealthVisualizerApp.jsx
// FINAL VERSION WITH NOTIFICATION FETCHING

import React, { useState, useMemo, useEffect } from 'react';
import { api } from './api'; // Import our helper
import HealthVisualizer from './HealthVisualizer.jsx';
import InfoPanel from './InfoPanel.jsx';
import Header from './Header.jsx';
import ChatPanel from './ChatPanel.jsx';
import './App.css';

const MAX_YEARS = 30;

export default function HealthVisualizerApp({ onLogout }) {
  const [user, setUser] = useState(null);
  const [lifestyle, setLifestyle] = useState({ Lungs: 0, Liver: 0, Heart: 0 });
  const [timeProjection, setTimeProjection] = useState(0);
  const [selectedOrgan, setSelectedOrgan] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isChatVisible, setIsChatVisible] = useState(false);
  
  // --- NEW: State for notifications ---
  const [notifications, setNotifications] = useState([]);

  // This single useEffect now handles fetching ALL initial user data
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        // Fetch user data and notifications in parallel for speed
        const [userData, notificationsData] = await Promise.all([
          api.get('/users/me/'),
          api.get('/notifications')
        ]);
        setUser(userData);
        setNotifications(notificationsData);
      } catch (error) {
        console.error("Failed to fetch initial data:", error);
        onLogout(); // If any essential data fails, log out
      }
    };
    fetchInitialData();
  }, [onLogout]);

  // Function to update the state when a notification is read
  const handleMarkAsRead = (notificationId) => {
    setNotifications(prev =>
      prev.map(n => (n.id === notificationId ? { ...n, is_read: true } : n))
    );
  };

  // ... (all other handler functions are the same) ...
  const handleLifestyleChange = (organ, value) => { setLifestyle(prevState => ({ ...prevState, [organ]: parseFloat(value) })); };
  const projectedStress = useMemo(() => { const timeFactor = timeProjection / MAX_YEARS; return { Lungs: lifestyle.Lungs * timeFactor, Liver: lifestyle.Liver * timeFactor, Heart: lifestyle.Heart * timeFactor }; }, [lifestyle, timeProjection]);
  const handleOrganClick = async (organName) => { /* ... unchanged ... */ };
  const handleClosePanel = () => { setSelectedOrgan(null); setPrediction(null); };
  const lifestyleDataForChat = { smokingIntensity: lifestyle.Lungs, alcoholIntensity: lifestyle.Liver, dietIntensity: lifestyle.Heart, timeProjection: timeProjection };

  return (
    <div className="app-container">
      {/* Pass notifications and the handler down to the Header */}
      <Header
        userEmail={user ? user.email : 'Loading...'}
        onLogout={onLogout}
        notifications={notifications}
        onMarkAsRead={handleMarkAsRead}
      />
      
      <main className="main-content">
        {/* ... (The rest of the JSX is the same as before) ... */}
        <div className="ui-panel"><h1>Health Controls</h1><div className="slider-group"><label>Smoking Intensity (Lungs)</label><input type="range" min="0" max="1" step="0.01" value={lifestyle.Lungs} onChange={(e) => handleLifestyleChange('Lungs', e.target.value)} /></div><div className="slider-group"><label>Alcohol Intensity (Liver)</label><input type="range" min="0" max="1" step="0.01" value={lifestyle.Liver} onChange={(e) => handleLifestyleChange('Liver', e.target.value)} /></div><div className="slider-group"><label>Poor Diet Intensity (Heart)</label><input type="range" min="0" max="1" step="0.01" value={lifestyle.Heart} onChange={(e) => handleLifestyleChange('Heart', e.target.value)} /></div><div className="slider-group time-slider"><label>Time Projection: {timeProjection} Years</label><input type="range" min="0" max={MAX_YEARS} step="1" value={timeProjection} onChange={(e) => setTimeProjection(parseInt(e.target.value))} /></div></div>
        <button className="chat-fab" onClick={() => setIsChatVisible(true)}>Chat with Twin</button>
        <div className="canvas-container"><HealthVisualizer organStress={projectedStress} onOrganClick={handleOrganClick} /></div>
        {selectedOrgan && (<InfoPanel organName={selectedOrgan} stressLevel={projectedStress[selectedOrgan]} onClose={handleClosePanel} prediction={prediction}/>)}
        <ChatPanel lifestyleData={lifestyleDataForChat} isVisible={isChatVisible} onClose={() => setIsChatVisible(false)}/>
      </main>
    </div>
  );
}