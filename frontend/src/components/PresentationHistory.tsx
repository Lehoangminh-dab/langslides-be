import React from 'react';
import { Presentation, Download, ExternalLink, Calendar, Layers } from 'lucide-react';
import { formatDistanceToNow, format } from 'date-fns';

interface PresentationItem {
  id: number;
  file_path: string;
  template: string;
  slide_count: number;
  created_at: string;
  gdrive_link: string | null;
  download_count: number;
}

interface PresentationHistoryProps {
  presentations: PresentationItem[];
}

export const PresentationHistory: React.FC<PresentationHistoryProps> = ({ presentations }) => {
  if (presentations.length === 0) {
    return (
      <div className="text-center p-8 bg-gray-50 rounded-lg">
        <p className="text-gray-500">You haven't generated any presentations yet.</p>
      </div>
    );
  }

  const getFileName = (path: string) => {
    const parts = path.split(/[\/\\]/);
    return parts[parts.length - 1];
  };

  const downloadPresentation = (sessionId: string) => {
    window.open(`http://localhost:8000/api/download?session_id=${sessionId}`, '_blank');
  };

  return (
    <div className="bg-white shadow overflow-hidden sm:rounded-lg">
      <ul className="divide-y divide-gray-200">
        {presentations.map((presentation) => (
          <li key={presentation.id}>
            <div className="px-4 py-4 sm:px-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <Presentation className="h-6 w-6 text-green-500" />
                  </div>
                  <p className="ml-3 text-sm font-medium text-green-600 truncate">
                    {getFileName(presentation.file_path)}
                  </p>
                </div>
                <div className="flex-shrink-0 flex">
                  <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                    {presentation.template} Template
                  </span>
                </div>
              </div>
              
              <div className="mt-2 sm:flex sm:justify-between">
                <div className="sm:flex items-center">
                  <div className="flex items-center text-sm text-gray-500">
                    <Layers size={16} className="mr-1.5 flex-shrink-0" />
                    <p>{presentation.slide_count} slides</p>
                  </div>
                  <div className="mt-2 flex items-center text-sm text-gray-500 sm:mt-0 sm:ml-6">
                    <Download size={16} className="mr-1.5 flex-shrink-0" />
                    <p>Downloaded {presentation.download_count} time{presentation.download_count !== 1 ? 's' : ''}</p>
                  </div>
                </div>
                <div className="mt-2 flex items-center text-sm text-gray-500 sm:mt-0">
                  <Calendar size={16} className="mr-1.5 flex-shrink-0" />
                  <p>{format(new Date(presentation.created_at), 'MMM d, yyyy')}</p>
                </div>
              </div>
              
              <div className="mt-3 flex space-x-3">
                <button
                  onClick={() => downloadPresentation(presentation.session_id)}
                  className="inline-flex items-center px-3 py-1.5 border border-green-300 text-xs font-medium rounded-md text-green-700 bg-green-50 hover:bg-green-100"
                >
                  <Download size={14} className="mr-1.5" /> Download
                </button>
                {presentation.gdrive_link && (
                  <a
                    href={presentation.gdrive_link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center px-3 py-1.5 border border-blue-300 text-xs font-medium rounded-md text-blue-700 bg-blue-50 hover:bg-blue-100"
                  >
                    <ExternalLink size={14} className="mr-1.5" /> View on Google Drive
                  </a>
                )}
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};