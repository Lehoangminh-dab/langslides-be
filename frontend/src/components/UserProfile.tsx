import React from 'react';
import { useApi } from '../context/ApiContext';
import { useAuth } from '../context/AuthContext';
import { User, FileText, MessageSquare, Presentation, LogOut } from 'lucide-react';

export const UserProfile: React.FC = () => {
  const { userProfile, pdfUploads, presentations, sessions } = useApi();
  const { logout } = useAuth();
  
  if (!userProfile) {
    return (
      <div className="p-4 bg-gray-50 rounded-lg flex items-center justify-center">
        <p className="text-gray-500">User profile not available</p>
      </div>
    );
  }
  
  return (
    <div className="space-y-6">
      {/* User info section */}
      <div className="flex flex-col sm:flex-row justify-between">
        <div className="flex items-center space-x-4">
          {userProfile.picture_url ? (
            <img 
              src={userProfile.picture_url} 
              alt={userProfile.name || 'User'} 
              className="w-16 h-16 rounded-full"
            />
          ) : (
            <div className="w-16 h-16 rounded-full bg-blue-100 flex items-center justify-center">
              <User size={32} className="text-blue-500" />
            </div>
          )}
          <div>
            <h2 className="text-xl font-semibold text-gray-800">{userProfile.name || 'User'}</h2>
            <p className="text-gray-500">{userProfile.email || ''}</p>
            <p className="text-xs text-gray-400">Member since {new Date(userProfile.created_at).toLocaleDateString()}</p>
          </div>
        </div>
        
        <div className="mt-4 sm:mt-0">
          <button 
            onClick={(e) => {
              e.preventDefault();
              logout();
              // No need to add redirect here since we've added it to the main logout function
            }} 
            className="flex items-center px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50 text-gray-700 transition"
          >
            <LogOut size={16} className="mr-2" />
            Sign out
          </button>
        </div>
      </div>
      
      {/* Stats boxes */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-blue-800">Uploaded PDFs</h3>
            <FileText className="h-5 w-5 text-blue-500" />
          </div>
          <p className="text-2xl font-bold text-blue-800 mt-2">{pdfUploads?.length || 0}</p>
        </div>
        
        <div className="bg-green-50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-green-800">Presentations</h3>
            <Presentation className="h-5 w-5 text-green-500" />
          </div>
          <p className="text-2xl font-bold text-green-800 mt-2">{presentations?.length || 0}</p>
        </div>
        
        <div className="bg-purple-50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-purple-800">Chat Sessions</h3>
            <MessageSquare className="h-5 w-5 text-purple-500" />
          </div>
          <p className="text-2xl font-bold text-purple-800 mt-2">{sessions?.length || 0}</p>
        </div>
      </div>
    </div>
  );
};