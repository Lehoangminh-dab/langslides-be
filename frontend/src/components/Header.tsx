import React from 'react';
import { Menu, Presentation } from 'lucide-react';
import { useApi } from '../hooks/useApi';

interface HeaderProps {
  onToggleSidebar: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onToggleSidebar }) => {
  const { downloadUrl, downloadPresentation, loading } = useApi();
  
  return (
    <header className="bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-md z-10">
      <div className="container mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center">
          <button
            className="mr-3 md:hidden"
            onClick={onToggleSidebar}
            aria-label="Toggle sidebar"
          >
            <Menu size={24} />
          </button>
          <div className="flex items-center gap-2">
            <Presentation size={28} className="text-white" />
            <h1 className="text-xl font-bold">PowerPoint Generator</h1>
          </div>
        </div>
        
        {downloadUrl && (
          <button
            onClick={downloadPresentation}
            disabled={loading}
            className="bg-white text-blue-600 px-4 py-2 rounded-md font-medium text-sm hover:bg-blue-50 transition duration-200 flex items-center gap-2 disabled:opacity-50"
          >
            <Presentation size={16} />
            Download Presentation
          </button>
        )}
      </div>
    </header>
  );
};