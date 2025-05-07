import React, { useRef } from 'react';
import ChatInterface from './ChatInterface';
import MessageBubble from './MessageBubble';

const ChatArea: React.FC = () => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messages = []; // Replace with actual messages data

  return (
    <div className="flex flex-col space-y-4 max-w-4xl mx-auto py-4">
      {messages.map((message, index) => (
        <MessageBubble 
          key={index}
          isUser={message[0] !== null}
          message={message[0] || message[1] || ""}
        />
      ))}
      {/* Add ref to last message for scrolling */}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default ChatArea;