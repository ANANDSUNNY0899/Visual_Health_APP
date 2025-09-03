// // src/ChatPanel.jsx
// import React, { useState, useRef, useEffect } from 'react';
// import './ChatPanel.css';

// export default function ChatPanel({ lifestyleData, isVisible, onClose }) {
//   const [messages, setMessages] = useState([
//     { sender: 'ai', text: 'Hello! I am your Digital Twin assistant. Ask me anything about your health simulation.' }
//   ]);
//   const [input, setInput] = useState('');
//   const [isLoading, setIsLoading] = useState(false);
//   const messagesEndRef = useRef(null);
  
//   // --- NEW: State to store the current chat session ID ---
//   const [sessionId, setSessionId] = useState(null);

//   const scrollToBottom = () => {
//     messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
//   };

//   // This effect now runs when the panel becomes visible
//   useEffect(() => {
//     // If the panel is visible and we don't have a session ID yet, create one.
//     if (isVisible && !sessionId) {
//       const createNewSession = async () => {
//         try {
//           const token = localStorage.getItem('visuhealth_token');
//           const response = await fetch('http://127.0.0.1:8000/chat/sessions', {
//             method: 'POST',
//             headers: { 'Authorization': `Bearer ${token}` }
//           });
//           const data = await response.json();
//           setSessionId(data.id); // Save the new session ID
//         } catch (error) {
//           console.error("Failed to create chat session:", error);
//         }
//       };
//       createNewSession();
//     }
//     scrollToBottom();
//   }, [isVisible, sessionId]); // Re-run if visibility changes

//   useEffect(scrollToBottom, [messages]);


//   const handleSendMessage = async (e) => {
//     e.preventDefault();
//     if (!input.trim() || isLoading || !sessionId) return; // Don't send if no session

//     const userMessage = { sender: 'user', text: input };
//     setMessages(prev => [...prev, userMessage]);
//     setInput('');
//     setIsLoading(true);

//     try {
//       const token = localStorage.getItem('visuhealth_token');
//       // --- THE FIX: Use the new, correct endpoint with the session ID ---
//       const response = await fetch(`http://127.0.0.1:8000/chat/sessions/${sessionId}`, {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//           'Authorization': `Bearer ${token}`
//         },
//         body: JSON.stringify({
//           question: input,
//           lifestyle_data: lifestyleData,
//         }),
//       });

//       if (!response.ok) {
//         throw new Error('Failed to get a response from the server.');
//       }

//       const data = await response.json();
//       const aiMessage = { sender: 'ai', text: data.text }; // The response is now a ChatMessageSchema
//       setMessages(prev => [...prev, aiMessage]);

//     } catch (error) {
//       console.error("Chat error:", error);
//       const errorMessage = { sender: 'ai', text: 'Sorry, I encountered an error. Please try again.' };
//       setMessages(prev => [...prev, errorMessage]);
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   if (!isVisible) return null;

//   return (
//     <div className="chat-panel">
//       {/* ... (The rest of the JSX is the same) ... */}
//       <div className="chat-header">
//         <h3>Chat with Your Twin</h3>
//         <button onClick={onClose} className="chat-close-btn">×</button>
//       </div>
//       <div className="chat-messages">
//         {messages.map((msg, index) => (
//           <div key={index} className={`message ${msg.sender}`}>
//             <p>{msg.text}</p>
//           </div>
//         ))}
//         {isLoading && (
//           <div className="message ai">
//             <div className="typing-indicator">
//               <span></span><span></span><span></span>
//             </div>
//           </div>
//         )}
//         <div ref={messagesEndRef} />
//       </div>
//       <form className="chat-input-form" onSubmit={handleSendMessage}>
//         <input
//           type="text"
//           value={input}
//           onChange={(e) => setInput(e.target.value)}
//           placeholder="Ask a health question..."
//           disabled={isLoading || !sessionId}
//         />
//         <button type="submit" disabled={isLoading || !sessionId}>Send</button>
//       </form>
//     </div>
//   );
// }   




// src/ChatPanel.jsx
// FINAL, ADVANCED VERSION WITH HISTORY, LANGUAGE, & STREAMING PREPARATION

import React, { useState, useRef, useEffect } from 'react';
import { api } from './api';
import './ChatPanel.css';

export default function ChatPanel({ lifestyleData, isVisible, onClose }) {
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [language, setLanguage] = useState('English'); // Language state
  const messagesEndRef = useRef(null);

  // Effect to fetch all chat sessions when the panel becomes visible
  useEffect(() => {
    if (isVisible) {
      const fetchSessions = async () => {
        try {
          const data = await api.get('/chat/sessions');
          setSessions(data);
          // If there are sessions, automatically select the most recent one
          if (data.length > 0) {
            selectSession(data[0].id);
          }
        } catch (error) {
          console.error("Failed to fetch sessions:", error);
        }
      };
      fetchSessions();
    }
  }, [isVisible]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(scrollToBottom, [messages]);

  // Function to load messages for a selected session
  const selectSession = async (sessionId) => {
    setActiveSessionId(sessionId);
    setMessages([]); // Clear old messages
    setIsLoading(true);
    try {
      const data = await api.get(`/chat/sessions/${sessionId}`);
      setMessages(data.map(msg => ({ sender: msg.sender, text: msg.text })));
    } catch (error) {
      console.error("Failed to fetch messages:", error);
    } finally {
      setIsLoading(false);
    }
  };

  // Function to create a new chat session
  const handleNewChat = async () => {
    try {
      const newSession = await api.post('/chat/sessions');
      setSessions(prev => [newSession, ...prev]);
      setActiveSessionId(newSession.id);
      setMessages([{ sender: 'ai', text: 'Hello! How can I help you today?' }]);
    } catch (error) {
      console.error("Failed to create new session:", error);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading || !activeSessionId) return;

    const userMessage = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const aiMessageData = await api.post(`/chat/sessions/${activeSessionId}`, {
        question: input,
        lifestyle_data: lifestyleData,
        language: language, // Send the selected language
      });
      const aiMessage = { sender: 'ai', text: aiMessageData.text };
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error("Chat error:", error);
      const errorMessage = { sender: 'ai', text: 'Sorry, I encountered an error. Please try again.' };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  if (!isVisible) return null;

  return (
    <div className="chat-panel">
      <div className="sidebar">
        <button className="new-chat-btn" onClick={handleNewChat}>+ New Chat</button>
        <div className="session-list">
          {sessions.map(session => (
            <div
              key={session.id}
              className={`session-item ${session.id === activeSessionId ? 'active' : ''}`}
              onClick={() => selectSession(session.id)}
            >
              {session.title}
            </div>
          ))}
        </div>
      </div>
      <div className="main-chat-area">
        <div className="chat-header">
          <div className="language-selector">
            <label htmlFor="lang">Language:</label>
            <select id="lang" value={language} onChange={(e) => setLanguage(e.target.value)}>
              <option value="English">English</option>
              <option value="Hindi">Hindi</option>
            </select>
          </div>
          <button onClick={onClose} className="chat-close-btn">×</button>
        </div>
        <div className="chat-messages">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}><p>{msg.text}</p></div>
          ))}
          {isLoading && <div className="message ai"><div className="typing-indicator"><span></span><span></span><span></span></div></div>}
          <div ref={messagesEndRef} />
        </div>
        <form className="chat-input-form" onSubmit={handleSendMessage}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a health question..."
            disabled={isLoading || !activeSessionId}
          />
          <button type="submit" disabled={isLoading || !activeSessionId}>Send</button>
        </form>
      </div>
    </div>
  );
}