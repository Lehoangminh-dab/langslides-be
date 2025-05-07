import React, { useState, useEffect } from 'react';
import { useApi } from '../context/ApiContext';
import { useAuth } from '../context/AuthContext';
import ChatInterface from './ChatInterface';
import Dashboard from './Dashboard';
import Sidebar from './Sidebar';
import ConnectionErrorMessage from './ConnectionErrorMessage';
import { User, Settings, X } from 'lucide-react';

const Layout: React.FC = () => {
  const { apiConnected, initialLoading } = useApi();
  const { isAuthenticated } = useAuth();
  const [showDashboard, setShowDashboard] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  
  useEffect(() => {
    const setVh = () => {
      const vh = window.innerHeight * 0.01;
      document.documentElement.style.setProperty('--vh', `${vh}px`);
    };
    
    // Set initial value
    setVh();
    
    // Update on resize, orientation change, and navigation events
    window.addEventListener('resize', setVh);
    window.addEventListener('orientationchange', setVh);
    window.addEventListener('popstate', setVh);
    
    // Create a mutation observer to detect DOM changes
    const observer = new MutationObserver(setVh);
    observer.observe(document.body, { childList: true, subtree: true });
    
    return () => {
      window.removeEventListener('resize', setVh);
      window.removeEventListener('orientationchange', setVh);
      window.removeEventListener('popstate', setVh);
      observer.disconnect();
    };
  }, []);
  
  // Only show loading state for initial app loading
  if (initialLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-blue-500 border-solid"></div>
      </div>
    );
  }
  
  // Show API connection error
  if (!apiConnected) {
    return <ConnectionErrorMessage />;
  }
  
  // Show main app
  return (
    <div className="flex h-screen bg-gray-100 relative">
      {/* Main content area - takes full width */}
      <div className="flex-1 flex flex-col h-screen min-w-0">
        {/* Header - compact on small screens */}
        <header className="sticky top-0 z-10 bg-white border-b shadow-sm w-full">
          <div className="px-3 sm:px-4 py-2 sm:py-3 flex items-center max-w-screen-2xl mx-auto w-full">
            <h1 className="text-lg sm:text-xl font-semibold text-gray-800 whitespace-nowrap overflow-hidden text-ellipsis">
              AI Presentation Generator
            </h1>
            
            {/* Action buttons container */}
            <div className="ml-auto flex items-center space-x-2">
              {/* Settings button */}
              {isAuthenticated && (
                <button 
                  onClick={() => setShowSettings(true)}
                  className="p-2 rounded-full hover:bg-gray-100 text-gray-500"
                  aria-label="Open settings"
                >
                  <Settings size={20} />
                </button>
              )}
              
              {/* Dashboard button */}
              {isAuthenticated && (
                <button 
                  onClick={() => setShowDashboard(!showDashboard)}
                  className="p-2 rounded-full hover:bg-gray-100 text-gray-500"
                  aria-label="Toggle dashboard"
                >
                  <User size={20} />
                </button>
              )}
            </div>
          </div>
        </header>
        
        {/* Chat area - takes full width */}
        <div className="flex-1 overflow-hidden relative flex flex-col">
          <ChatInterface />
        </div>

        {/* Settings Panel - positioned as overlay with higher z-index */}
        {showSettings && (
          <div className="fixed inset-0 z-50 bg-black bg-opacity-50 flex items-center justify-center p-2 sm:p-4">
            <div className="bg-white rounded-lg shadow-xl w-full max-w-[95vw] sm:max-w-[90vw] md:max-w-3xl max-h-[90vh] overflow-hidden flex flex-col">
              <div className="flex justify-between items-center border-b p-3 sm:p-4">
                <h2 className="text-lg sm:text-xl font-semibold">Settings</h2>
                <button 
                  onClick={() => setShowSettings(false)}
                  className="p-1 rounded-full hover:bg-gray-100 text-gray-500"
                >
                  <X size={20} />
                </button>
              </div>
              
              <div className="overflow-y-auto flex-1">
                <Sidebar isInSettingsPanel={true} />
              </div>
            </div>
          </div>
        )}
        
        {/* Dashboard overlay - with higher z-index */}
        {showDashboard && (
          <div className="fixed inset-0 z-50 bg-black bg-opacity-50 flex items-center justify-center p-2 sm:p-4">
            <div className="bg-white rounded-lg shadow-xl w-full max-w-[95vw] sm:max-w-[90vw] md:max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
              <div className="flex justify-between items-center border-b p-3 sm:p-4">
                <h2 className="text-lg sm:text-xl font-semibold">Your Dashboard</h2>
                <button 
                  onClick={() => setShowDashboard(false)}
                  className="p-1 rounded-full hover:bg-gray-100 text-gray-500"
                >
                  <X size={20} />
                </button>
              </div>
              <div className="overflow-y-auto flex-1 p-3 sm:p-4">
                <Dashboard key={`dashboard-${showDashboard}`} />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Layout;