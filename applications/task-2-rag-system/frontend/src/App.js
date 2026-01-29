import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, Upload, Trash2, MessageCircle, Plus } from 'lucide-react';
import ChatMessage from './components/ChatMessage';
import DocumentList from './components/DocumentList';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [activeTab, setActiveTab] = useState('chat');
  const [showSidebar, setShowSidebar] = useState(true);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const newId = `conv-${Date.now()}`;
    setConversationId(newId);
  }, []);

  useEffect(() => {
    if (activeTab === 'documents') {
      loadDocuments();
    }
  }, [activeTab]);

  const loadDocuments = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/documents`);
      setDocuments(response.data.documents);
    } catch (error) {
      console.error('Error loading documents:', error);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    // Add user message
    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString(),
    };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/query`, {
        query: input,
        conversation_id: conversationId,
        top_k: 5,
      });

      const assistantMessage = {
        role: 'assistant',
        content: response.data.answer,
        sources: response.data.sources,
        confidence: response.data.confidence,
        timestamp: response.data.timestamp,
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, there was an error processing your query. Please try again.',
        error: true,
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleUploadDocument = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      setLoading(true);
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      // Reload documents
      await loadDocuments();

      // Add system message
      const systemMessage = {
        role: 'system',
        content: `✅ Document "${file.name}" uploaded successfully`,
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, systemMessage]);
    } catch (error) {
      console.error('Error uploading document:', error);
      const errorMessage = {
        role: 'system',
        content: `❌ Error uploading document: ${error.response?.data?.detail || error.message}`,
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleClearConversation = async () => {
    try {
      await axios.post(`${API_BASE_URL}/conversation/${conversationId}/clear`);
      setMessages([]);
      setConversationId(`conv-${Date.now()}`);
    } catch (error) {
      console.error('Error clearing conversation:', error);
    }
  };

  const handleDeleteDocument = async (docId) => {
    try {
      await axios.delete(`${API_BASE_URL}/documents/${docId}`);
      await loadDocuments();
    } catch (error) {
      console.error('Error deleting document:', error);
    }
  };

  return (
    <div className="app">
      {/* Sidebar */}
      {showSidebar && (
        <div className="sidebar">
          <div className="sidebar-header">
            <h1>🤖 RAG System</h1>
          </div>

          <nav className="sidebar-nav">
            <button
              className={`nav-item ${activeTab === 'chat' ? 'active' : ''}`}
              onClick={() => { setActiveTab('chat'); setShowSidebar(false); }}
            >
              <MessageCircle size={20} />
              Chat
            </button>
            <button
              className={`nav-item ${activeTab === 'documents' ? 'active' : ''}`}
              onClick={() => setActiveTab('documents')}
            >
              <Upload size={20} />
              Documents
            </button>
          </nav>

          <div className="sidebar-footer">
            <button className="btn-secondary" onClick={handleClearConversation}>
              <Plus size={16} />
              New Conversation
            </button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="main-content">
        {activeTab === 'chat' && (
          <>
            {/* Chat Area */}
            <div className="chat-container">
              <div className="chat-header">
                <button
                  className="toggle-sidebar"
                  onClick={() => setShowSidebar(!showSidebar)}
                >
                  ☰
                </button>
                <h2>Chat with RAG System</h2>
              </div>

              <div className="messages-area">
                {messages.length === 0 ? (
                  <div className="empty-state">
                    <h3>Welcome to RAG System</h3>
                    <p>Upload documents and ask questions about them</p>
                  </div>
                ) : (
                  messages.map((msg, idx) => (
                    <ChatMessage key={idx} message={msg} />
                  ))
                )}
                {loading && <div className="loading-indicator">Thinking...</div>}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <form className="input-area" onSubmit={handleSendMessage}>
                <label htmlFor="document-upload" className="upload-btn">
                  <Upload size={20} />
                </label>
                <input
                  type="file"
                  id="document-upload"
                  onChange={handleUploadDocument}
                  disabled={loading}
                  style={{ display: 'none' }}
                />

                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask about your documents..."
                  disabled={loading}
                  className="message-input"
                />

                <button
                  type="submit"
                  disabled={loading || !input.trim()}
                  className="send-btn"
                >
                  <Send size={20} />
                </button>
              </form>
            </div>
          </>
        )}

        {activeTab === 'documents' && (
          <DocumentList
            documents={documents}
            onDelete={handleDeleteDocument}
            onBack={() => setActiveTab('chat')}
          />
        )}
      </div>
    </div>
  );
}
