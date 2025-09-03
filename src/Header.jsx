// src/Header.jsx
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import NotificationsPanel from './NotificationsPanel.jsx'; // Import the new panel
import './Header.css';

// The Header now receives the notifications and the handler function
export default function Header({ userEmail, onLogout, notifications, onMarkAsRead }) {
  const [isPanelVisible, setIsPanelVisible] = useState(false);

  // Calculate the number of unread notifications
  const unreadCount = notifications ? notifications.filter(n => !n.is_read).length : 0;

  return (
    <header className="app-header">
      <div className="header-title">
        <h1>VisuHealth</h1>
        <span>Your Digital Twin</span>
      </div>
      <div className="header-user-info">
        <Link to="/reports" className="header-link">Manage Reports</Link>
        <Link to="/logbook" className="header-link">My Health Log</Link>
        <Link to="/reports" className="header-link">Manage Reports</Link>
        <span>{userEmail}</span>
        
        {/* --- NEW Notification Bell --- */}
        <div className="notification-bell" onClick={() => setIsPanelVisible(!isPanelVisible)}>
      {/* A more elegant, outlined bell icon */}
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
        <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
      </svg>
      {unreadCount > 0 && (
        <div className="notification-badge">{unreadCount}</div>
      )}
    </div>
        
        <button onClick={onLogout} className="header-logout-btn">Logout</button>
      </div>

      {/* Conditionally render the panel */}
      {isPanelVisible && (
        <NotificationsPanel
          notifications={notifications}
          onMarkAsRead={onMarkAsRead}
          onClose={() => setIsPanelVisible(false)}
        />
      )}
    </header>
  );
}