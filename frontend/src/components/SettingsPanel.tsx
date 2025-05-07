import React from 'react';
import { X } from 'lucide-react';
import Sidebar from './Sidebar';

interface SettingsPanelProps {
  onClose: () => void;
}

const SettingsPanel: React.FC<SettingsPanelProps> = ({ onClose }) => {
  return (
    <div className="fixed inset-0 z-30 bg-black bg-opacity-50 flex items-center justify-center p-2 sm:p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-[95vw] sm:max-w-[90vw] md:max-w-3xl max-h-[90vh] overflow-hidden flex flex-col">
        <div className="flex justify-between items-center border-b p-3 sm:p-4">
          <h2 className="text-lg sm:text-xl font-semibold">Settings</h2>
          <button 
            onClick={onClose}
            className="p-1 rounded-full hover:bg-gray-100 text-gray-500"
          >
            <X size={20} />
          </button>
        </div>
        
        <div className="overflow-y-auto flex-1 p-2">
          <Sidebar isInSettingsPanel={true} />
        </div>
      </div>
    </div>
  );
};

export default SettingsPanel;