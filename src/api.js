// src/api.js
// FINAL, CORRECTED VERSION

const API_BASE_URL = 'http://127.0.0.1:8000';

const request = async (endpoint, options = {}) => {
  const token = localStorage.getItem('visuhealth_token');
  const headers = {
    ...options.headers,
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  
  const response = await fetch(`${API_BASE_URL}${endpoint}`, { ...options, headers });

  if (!response.ok) {
    try {
      const errorData = await response.json();
      throw new Error(errorData.detail || `Request failed with status ${response.status}`);
    } catch {
      throw new Error(`Request failed with status ${response.status}`);
    }
  }

  if (response.status === 204) {
    return null;
  }
  
  return response.json();
};

export const api = {
  get: (endpoint) => request(endpoint),
  
  post: (endpoint, body) => request(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }),

  // --- THE FIX: Add the missing 'postForm' method ---
  // This is used for file uploads with FormData. Notice it does NOT set
  // the 'Content-Type' header, as the browser must do that automatically.
  postForm: (endpoint, formData) => request(endpoint, {
    method: 'POST',
    body: formData,
  }),
  // --- NEW: Add the delete method ---
  delete: (endpoint) => request(endpoint, {
    method: 'DELETE',
  }),
};