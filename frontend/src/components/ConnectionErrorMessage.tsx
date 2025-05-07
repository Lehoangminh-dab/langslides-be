import React from 'react';

const ConnectionErrorMessage: React.FC = () => {
  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-50">
      <div className="p-8 bg-white rounded-lg shadow-lg max-w-md text-center">
        <h2 className="text-2xl font-bold text-red-600 mb-4">Connection Error</h2>
        <p className="mb-6">
          Cannot connect to the API server. Please make sure the backend is running at http://localhost:8000.
        </p>
        <button 
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600" 
          onClick={() => window.location.reload()}
        >
          Retry Connection
        </button>
      </div>
    </div>
  );
};

export default ConnectionErrorMessage;