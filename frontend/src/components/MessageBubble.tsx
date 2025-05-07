import React from 'react';
import { User, Bot, Copy, Check } from 'lucide-react';
import { format } from 'date-fns';

interface MessageBubbleProps {
  message: string | null;
  isUser: boolean;
  userMessage?: string;
  timestamp?: Date;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  isUser,
  userMessage,
  timestamp = new Date()
}) => {
  const [copied, setCopied] = React.useState(false);
  
  if (!message && !isUser) return null;
  
  const handleCopy = () => {
    const textToCopy = isUser ? userMessage : message;
    if (textToCopy) {
      navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4 group`}>
      <div 
        className={`chat-message-container relative rounded-lg shadow-sm p-4 transition-all duration-200
          ${isUser 
            ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-br-none' 
            : 'bg-white border border-gray-200 rounded-bl-none'}`}
      >
        <div className="flex space-x-3">
          <div className="flex-shrink-0">
            {isUser ? (
              <div className="h-10 w-10 rounded-full bg-blue-700 flex items-center justify-center">
                <User className="h-5 w-5 text-white" />
              </div>
            ) : (
              <div className="h-10 w-10 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center">
                <Bot className="h-5 w-5 text-white" />
              </div>
            )}
          </div>
          <div className="flex-1 overflow-hidden">
            <div className="flex flex-col">
              {/* Message content */}
              <div className={`${isUser ? 'text-white' : 'text-gray-800'} whitespace-pre-wrap break-words`}>
                {isUser ? userMessage : message}
              </div>
              
              {/* Timestamp */}
              <div className={`message-timestamp mt-2 ${isUser ? 'text-blue-100' : 'text-gray-400'}`}>
                {format(timestamp, 'h:mm a')}
              </div>
            </div>
          </div>
        </div>
        
        {/* Copy button that appears on hover */}
        <button 
          onClick={handleCopy}
          className={`absolute top-2 right-2 p-1 rounded-full transition-opacity duration-200 
            ${isUser ? 'bg-blue-800 hover:bg-blue-900' : 'bg-gray-100 hover:bg-gray-200'} 
            opacity-0 group-hover:opacity-100`}
          title="Copy message"
        >
          {copied ? (
            <Check size={14} className={isUser ? 'text-blue-100' : 'text-green-500'} />
          ) : (
            <Copy size={14} className={isUser ? 'text-blue-100' : 'text-gray-500'} />
          )}
        </button>
      </div>
    </div>
  );
};