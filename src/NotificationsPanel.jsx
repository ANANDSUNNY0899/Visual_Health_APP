// src/NotificationsPanel.jsx

import React from 'react';
import { api } from './api';
import './NotificationsPanel.css';

export default function NotificationsPanel({ notifications, onMarkAsRead, onClose }) {
  if (!notifications) return null;

  const handleNotificationClick = async (notification) => {
    if (!notification.is_read) {
      try {
        // Call the backend to mark it as read
        await api.post(`/notifications/${notification.id}/read`, {});
        // Tell the parent component to update the state
        onMarkAsRead(notification.id);
      } catch (error) {
        console.error("Failed to mark notification as read:", error);
      }
    }
  };

  return (
    <div className="notifications-panel">
      <div className="notifications-header">
        <h4>Your Health Insights</h4>
        <button onClick={onClose} className="close-panel-btn">×</button>
      </div>
      <div className="notifications-list">
        {notifications.length > 0 ? (
          notifications.map(notif => (
            <div
              key={notif.id}
              className={`notification-item ${!notif.is_read ? 'unread' : ''}`}
              onClick={() => handleNotificationClick(notif)}
            >
              <p>{notif.message}</p>
              <span className="timestamp">{new Date(notif.created_at).toLocaleString()}</span>
            </div>
          ))
        ) : (
          <div className="no-notifications">
            <p>No new insights at the moment. Check back later!</p>
          </div>
        )}
      </div>
    </div>
  );
}