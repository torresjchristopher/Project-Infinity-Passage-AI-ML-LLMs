import React from 'react';
import ReactMarkdown from 'react-markdown';

export default function ChatMessage({ message }) {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';

  return (
    <div className={`message ${message.role}`}>
      <div className="message-avatar">
        {isUser ? '👤' : isSystem ? 'ℹ️' : '🤖'}
      </div>
      <div className="message-content">
        <div className="message-text">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>

        {message.sources && message.sources.length > 0 && (
          <div className="sources">
            <strong>📚 Sources:</strong>
            <ul>
              {message.sources.map((source, idx) => (
                <li key={idx}>
                  <em>{source.filename}</em> (Chunk {source.chunk}, {(source.relevance * 100).toFixed(0)}% relevant)
                </li>
              ))}
            </ul>
          </div>
        )}

        {message.confidence && (
          <div className="confidence">
            Confidence: {(message.confidence * 100).toFixed(1)}%
          </div>
        )}

        {message.error && (
          <div className="message-error">⚠️ Error</div>
        )}
      </div>
    </div>
  );
}
