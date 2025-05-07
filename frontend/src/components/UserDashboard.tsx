import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import { FileHistory } from './FileHistory';
import { PresentationHistory } from './PresentationHistory';
import { SessionHistory } from './SessionHistory';
import { User, Database, FileText, Presentation } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

export const UserDashboard: React.FC = () => {
  const { isAuthenticated } = useAuth();
  const [activeTab, setActiveTab] = useState('uploads');
  const [userProfile, setUserProfile] = useState<any>(null);
  const [uploads, setUploads] = useState<any[]>([]);
  const [presentations, setPresentations] = useState<any[]>([]);
  const [sessions, setSessions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isAuthenticated) {
      fetchUserData();
    }
  }, [isAuthenticated]);

  const fetchUserData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Fetch all data in parallel
      const [profileRes, uploadsRes, presentationsRes, sessionsRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/api/user-profile`, { withCredentials: true }),
        axios.get(`${API_BASE_URL}/api/user-uploads`, { withCredentials: true }),
        axios.get(`${API_BASE_URL}/api/user-presentations`, { withCredentials: true }),
        axios.get(`${API_BASE_URL}/api/user-sessions`, { withCredentials: true }),
      ]);
      
      if (profileRes.data.success) {
        setUserProfile(profileRes.data.profile);
      }
      
      if (uploadsRes.data.success) {
        setUploads(uploadsRes.data.uploads);
      }
      
      if (presentationsRes.data.success) {
        setPresentations(presentationsRes.data.presentations);
      }
      
      if (sessionsRes.data.success) {
        setSessions(sessionsRes.data.sessions);
      }
      
    } catch (err) {
      console.error('Error fetching user data:', err);
      setError('Failed to load user data. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  if (!isAuthenticated) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center p-6 bg-white rounded-lg shadow">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Not Logged In</h2>
          <p className="text-gray-600">Please log in to view your dashboard.</p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-6 max-w-6xl">
      <h1 className="text-2xl font-bold text-gray-800 mb-6">Your Dashboard</h1>
      
      {/* User Profile Card */}
      {userProfile && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="flex items-center space-x-4">
            {userProfile.picture ? (
              <img 
                src={userProfile.picture} 
                alt={userProfile.name} 
                className="h-16 w-16 rounded-full"
              />
            ) : (
              <div className="h-16 w-16 rounded-full bg-blue-100 flex items-center justify-center">
                <User size={32} className="text-blue-500" />
              </div>
            )}
            <div>
              <h2 className="text-xl font-semibold">{userProfile.name}</h2>
              <p className="text-gray-600">{userProfile.email}</p>
            </div>
          </div>
          
          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            <div className="bg-blue-50 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-blue-800">Uploaded PDFs</h3>
                <FileText className="h-5 w-5 text-blue-500" />
              </div>
              <p className="text-2xl font-bold text-blue-800 mt-2">{uploads.length}</p>
            </div>
            
            <div className="bg-green-50 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-green-800">Presentations</h3>
                <Presentation className="h-5 w-5 text-green-500" />
              </div>
              <p className="text-2xl font-bold text-green-800 mt-2">{presentations.length}</p>
            </div>
            
            <div className="bg-purple-50 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-purple-800">Chat Sessions</h3>
                <Database className="h-5 w-5 text-purple-500" />
              </div>
              <p className="text-2xl font-bold text-purple-800 mt-2">{sessions.length}</p>
            </div>
          </div>
        </div>
      )}
      
      {/* Tabs Navigation */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="flex space-x-8">
          <button
            onClick={() => setActiveTab('uploads')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'uploads'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Uploaded PDFs
          </button>
          <button
            onClick={() => setActiveTab('presentations')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'presentations'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Generated Presentations
          </button>
          <button
            onClick={() => setActiveTab('sessions')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'sessions'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Chat History
          </button>
        </nav>
      </div>
      
      {/* Tab Content */}
      <div className="mt-6">
        {activeTab === 'uploads' && <FileHistory files={uploads} />}
        {activeTab === 'presentations' && <PresentationHistory presentations={presentations} />}
        {activeTab === 'sessions' && <SessionHistory sessions={sessions} />}
      </div>
    </div>
  );
};