import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

interface User {
  sub: string;
  name: string;
  email: string;
  picture: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: () => void;
  logout: () => void;
  checkAuth: () => Promise<boolean>;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  isAuthenticated: false,
  isLoading: true,
  login: () => {},
  logout: () => {},
  checkAuth: async () => false
});

export const useAuth = () => useContext(AuthContext);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  const API_BASE_URL = 'http://localhost:8000';

  const checkAuth = async () => {
    try {
      setIsLoading(true);
      const { data } = await axios.get(`${API_BASE_URL}/api/user`, { withCredentials: true });
      if (data.authenticated && data.user) {
        setUser(data.user);
        setIsAuthenticated(true);
        return true;
      } else {
        setUser(null);
        setIsAuthenticated(false);
        return false;
      }
    } catch (error) {
      console.error('Auth check failed:', error);
      setUser(null);
      setIsAuthenticated(false);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const login = () => {
    window.location.href = `${API_BASE_URL}/auth/login`;
  };

  const logout = async () => {
    try {
      await axios.get(`${API_BASE_URL}/api/logout`, { withCredentials: true });
      setUser(null);
      setIsAuthenticated(false);
      
      // Add this line to redirect to login page after successful logout
      window.location.href = '/login';
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  // Check auth status when the component mounts
  useEffect(() => {
    const checkAuthStatus = async () => {
      // Check for auth success in URL parameters
      const urlParams = new URLSearchParams(window.location.search);
      const authStatus = urlParams.get('auth');
      const userId = urlParams.get('userId');
      
      if (authStatus === 'success' && userId) {
        // Clear URL parameters
        window.history.replaceState({}, document.title, window.location.pathname);
        await checkAuth();
      } else {
        await checkAuth();
      }
    };
    
    checkAuthStatus();
  }, []);

  return (
    <AuthContext.Provider value={{ user, isAuthenticated, isLoading, login, logout, checkAuth }}>
      {children}
    </AuthContext.Provider>
  );
};