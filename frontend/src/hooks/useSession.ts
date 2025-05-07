import { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';
interface SessionData {
  sessionId: string | null;
  loading: boolean;
  error: string | null;
}

export const useSession = (): SessionData => {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const initSession = async () => {
      try {
        const { data } = await axios.post(`${API_BASE_URL}/api/session`);
        setSessionId(data.session_id);
        setLoading(false);
      } catch (error) {
        console.error('Error initializing session:', error);
        setError('Failed to initialize session');
        setLoading(false);
      }
    };
    
    initSession();
  }, []);
  
  return { sessionId, loading, error };
};