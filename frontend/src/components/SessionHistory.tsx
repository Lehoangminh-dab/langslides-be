import React from 'react';
import { MessageSquare, Calendar, AlertCircle } from 'lucide-react';
import { formatDistanceToNow, format } from 'date-fns';

interface Session {
  id: string;
  created_at: string;
  last_active: string;
  message_count: number;
}

interface SessionHistoryProps {
  sessions: Session[];
}

export const SessionHistory: React.FC<SessionHistoryProps> = ({ sessions }) => {
  if (!sessions || sessions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-8 bg-gray-50 rounded-lg">
        <AlertCircle className="h-12 w-12 text-gray-400 mb-4" />
        <p className="text-gray-500 text-center">No chat history found.</p>
        <p className="text-gray-400 text-sm text-center mt-2">
          Start a conversation to see your chat history.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white shadow overflow-hidden sm:rounded-lg">
      <ul className="divide-y divide-gray-200">
        {sessions.map((session) => (
          <li key={session.id}>
            <div className="px-4 py-4 sm:px-6 hover:bg-gray-50 transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <MessageSquare className="h-6 w-6 text-purple-500" />
                  </div>
                  <p className="ml-3 text-sm font-medium text-purple-600 truncate">
                    Session {session.id.substring(0, 8)}...
                  </p>
                </div>
                <div className="flex-shrink-0 flex">
                  <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-purple-100 text-purple-800">
                    {session.message_count} messages
                  </span>
                </div>
              </div>
              
              <div className="mt-2 sm:flex sm:justify-between">
                <div className="flex items-center text-sm text-gray-500">
                  <Calendar size={16} className="mr-1.5 flex-shrink-0" />
                  <p>
                    Created {formatDistanceToNow(new Date(session.created_at))} ago
                  </p>
                </div>
                <div className="mt-2 flex items-center text-sm text-gray-500 sm:mt-0">
                  <Calendar size={16} className="mr-1.5 flex-shrink-0" />
                  <p>
                    Last active {formatDistanceToNow(new Date(session.last_active))} ago
                  </p>
                </div>
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};