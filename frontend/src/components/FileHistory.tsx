import React from 'react';
import { FileText, Calendar, AlertCircle } from 'lucide-react';
import { formatDistanceToNow, format } from 'date-fns';

interface File {
  id: number;
  file_name: string;
  collection_name: string;
  created_at: string;
  file_hash?: string;
}

interface FileHistoryProps {
  files: File[];
}

export const FileHistory: React.FC<FileHistoryProps> = ({ files }) => {
  if (!files || files.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-8 bg-gray-50 rounded-lg">
        <AlertCircle className="h-12 w-12 text-gray-400 mb-4" />
        <p className="text-gray-500 text-center">You haven't uploaded any PDF files yet.</p>
        <p className="text-gray-400 text-sm text-center mt-2">
          Upload PDFs to enhance your presentations with relevant context.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white shadow overflow-hidden sm:rounded-lg">
      <ul className="divide-y divide-gray-200">
        {files.map((file) => (
          <li key={file.id}>
            <div className="px-4 py-4 sm:px-6 hover:bg-gray-50 transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <FileText className="h-6 w-6 text-blue-500" />
                  </div>
                  <p className="ml-3 text-sm font-medium text-blue-600 truncate">{file.file_name}</p>
                </div>
                <div className="flex-shrink-0 flex">
                  <p className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                    {file.collection_name}
                  </p>
                </div>
              </div>
              <div className="mt-2 sm:flex sm:justify-between">
                <div className="flex items-center text-sm text-gray-500">
                  <Calendar size={16} className="mr-1.5 flex-shrink-0" />
                  <p>
                    Uploaded {formatDistanceToNow(new Date(file.created_at))} ago
                    <span className="mx-1">•</span>
                    {format(new Date(file.created_at), 'MMM d, yyyy')}
                  </p>
                </div>
                {file.file_hash && (
                  <div className="mt-2 sm:mt-0 text-sm text-gray-500">
                    <div className="flex items-center">
                      <p className="text-xs font-mono">Hash: {file.file_hash.substring(0, 8)}...</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};