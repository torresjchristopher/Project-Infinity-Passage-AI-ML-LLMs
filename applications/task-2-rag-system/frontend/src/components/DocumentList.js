import React from 'react';
import { ArrowLeft, Trash2 } from 'lucide-react';

export default function DocumentList({ documents, onDelete, onBack }) {
  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="documents-container">
      <div className="documents-header">
        <button className="back-btn" onClick={onBack}>
          <ArrowLeft size={20} />
          Back to Chat
        </button>
        <h2>📄 Documents</h2>
      </div>

      {documents.length === 0 ? (
        <div className="empty-state">
          <p>No documents uploaded yet</p>
          <p>Go to chat and click the upload button to add documents</p>
        </div>
      ) : (
        <div className="documents-list">
          {documents.map((doc) => (
            <div key={doc.id} className="document-item">
              <div className="document-info">
                <h4>{doc.filename}</h4>
                <p>{formatSize(doc.size)}</p>
                <p className="document-date">
                  Uploaded: {formatDate(doc.uploaded_at)}
                </p>
              </div>
              <button
                className="delete-btn"
                onClick={() => onDelete(doc.id)}
                title="Delete document"
              >
                <Trash2 size={18} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
