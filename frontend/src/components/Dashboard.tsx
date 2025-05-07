import React, { useState, useEffect } from 'react';
import { FileHistory } from './FileHistory';
import { PresentationHistory } from './PresentationHistory';
import { SessionHistory } from './SessionHistory';
import { UserProfile } from './UserProfile';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/Tabs';
import { useApi } from '../context/ApiContext';

const Dashboard: React.FC = () => {
  const { userProfile, pdfUploads, presentations, sessions, fetchUserData, loadingUserData } = useApi();
  const [dataFetched, setDataFetched] = useState(false);
  
  // Only fetch data once when component is mounted
  useEffect(() => {
    if (!dataFetched) {
      fetchUserData();
      setDataFetched(true);
    }
    // Clean up function to prevent memory leaks
    return () => {
      // Any cleanup if needed
    };
  }, [dataFetched, fetchUserData]);
  
  if (loadingUserData) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="bg-white shadow rounded-lg overflow-hidden">
      <div className="p-6">
        <UserProfile />
      </div>
      
      <Tabs defaultValue="uploads" className="w-full">
        <div className="border-b">
          <TabsList className="flex">
            <TabsTrigger 
              value="uploads" 
              className="flex-1 py-3 px-4 border-b-2 border-transparent text-sm font-medium"
              activeClassName="border-blue-500 text-blue-600"
              inactiveClassName="text-gray-500 hover:text-gray-700 hover:border-gray-300"
            >
              Uploaded PDFs
            </TabsTrigger>
            <TabsTrigger 
              value="presentations"
              className="flex-1 py-3 px-4 border-b-2 border-transparent text-sm font-medium"
              activeClassName="border-blue-500 text-blue-600"
              inactiveClassName="text-gray-500 hover:text-gray-700 hover:border-gray-300"
            >
              Presentations
            </TabsTrigger>
            <TabsTrigger 
              value="sessions"
              className="flex-1 py-3 px-4 border-b-2 border-transparent text-sm font-medium"
              activeClassName="border-blue-500 text-blue-600"
              inactiveClassName="text-gray-500 hover:text-gray-700 hover:border-gray-300"
            >
              Chat History
            </TabsTrigger>
          </TabsList>
        </div>
        
        <div className="p-4">
          <TabsContent value="uploads">
            <FileHistory files={pdfUploads || []} />
          </TabsContent>
          
          <TabsContent value="presentations">
            <PresentationHistory presentations={presentations || []} />
          </TabsContent>
          
          <TabsContent value="sessions">
            <SessionHistory sessions={sessions || []} />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
};

export default Dashboard;