import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { ApiProvider } from './context/ApiContext';
import AppRoutes from './routes/AppRoutes';

function App() {
  return (
    <AuthProvider>
      <ApiProvider>
        <Router>
          <AppRoutes />
        </Router>
      </ApiProvider>
    </AuthProvider>
  );
}

export default App;