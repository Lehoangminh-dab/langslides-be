import React, { useState, useRef, useEffect } from 'react';
import { useApi } from '../context/ApiContext';
import { SendHorizontal, Presentation, Sparkles, Clock, RefreshCw, Zap } from 'lucide-react';
import { MessageBubble } from './MessageBubble';
import { format } from 'date-fns';

export const ChatInterface: React.FC = () => {
  const { 
    messages, 
    setMessages,
    generatingPresentation, 
    setGeneratingPresentation,
    generatePresentation, 
    downloadUrl, 
    downloadPresentation,
    clearHistory
  } = useApi();
  
  const [message, setMessage] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [downloadClicked, setDownloadClicked] = useState(false);
  const [startTime, setStartTime] = useState<Date | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Check if message is empty
    if (!message.trim() || generatingPresentation) return;
    
    // Recalculate viewport height when submitting
    const vh = window.innerHeight * 0.01;
    document.documentElement.style.setProperty('--vh', `${vh}px`);
    
    // Set start time for generating
    setStartTime(new Date());
    
    // Clear input field immediately
    const currentMessage = message.trim();
    setMessage("");
    
    // Update UI state - add user message to chat immediately
    setGeneratingPresentation(true);
    setMessages(prev => [...prev, [currentMessage, null]]);

    try {
      // This will update messages with server response
      await generatePresentation(currentMessage);
      
      // Focus input field after generation
      setTimeout(() => {
        const inputElement = document.getElementById('message-input');
        if (inputElement) inputElement.focus();
      }, 100);
    } catch (error) {
      console.error('Error in handleSubmit:', error);
    } finally {
      setStartTime(null);
    }
  };

  const handleDownload = () => {
    downloadPresentation();
    setDownloadClicked(true);
    
    // Reset download button after a delay
    setTimeout(() => {
      setDownloadClicked(false);
    }, 5000);
  };

  const handleClearChat = () => {
    if (window.confirm("Are you sure you want to clear the chat history?")) {
      clearHistory();
      setDownloadClicked(false);
    }
  };
  
  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Timer effect during generation
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (generatingPresentation && startTime) {
      interval = setInterval(() => {
        setElapsedTime(Math.floor((new Date().getTime() - startTime.getTime()) / 1000));
      }, 1000);
    } else {
      setElapsedTime(0);
    }
    
    return () => clearInterval(interval);
  }, [generatingPresentation, startTime]);

  // Format time as mm:ss
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  return (
    <div className="flex flex-col h-full relative">
      {/* Background pattern with subtle grid */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50 to-white z-0">
        <div className="absolute inset-0 bg-grid-pattern opacity-10"></div>
      </div>

      {/* Chat messages area - with improved responsive constraints */}
      <div className="flex-1 overflow-y-auto relative z-10 pb-16 sm:pb-20"> 
        <div className="max-w-4xl mx-auto px-3 sm:px-4 py-4 sm:py-6">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full py-10 sm:py-20">
              <div className="text-center p-5 sm:p-6 bg-white bg-opacity-80 rounded-lg shadow-sm border border-blue-100 max-w-md mx-auto">
                <div className="flex justify-center mb-4">
                  <div className="p-3 bg-blue-50 rounded-full">
                    <Sparkles className="h-12 w-12 text-blue-500" />
                  </div>
                </div>
                
                <h2 className="text-xl sm:text-2xl font-semibold text-gray-800">PowerPoint Generator</h2>
                <p className="mt-3 text-gray-600">Describe the presentation you want to create</p>
                <p className="mt-2 text-sm text-gray-500">For example: "Create a 10-slide presentation about renewable energy"</p>
                
                <div className="mt-6 space-y-3">
                  <SuggestionButton 
                    text="Create a 5-slide presentation about artificial intelligence"
                    onClick={(text) => {
                      setMessage(text);
                      setTimeout(() => {
                        const form = document.querySelector('form');
                        if (form) form.dispatchEvent(new Event('submit', { cancelable: true }));
                      }, 100);
                    }}
                  />
                  <SuggestionButton 
                    text="Make a business plan presentation with financial projections"
                    onClick={(text) => {
                      setMessage(text);
                      setTimeout(() => {
                        const form = document.querySelector('form');
                        if (form) form.dispatchEvent(new Event('submit', { cancelable: true }));
                      }, 100);
                    }}
                  />
                  <SuggestionButton 
                    text="Design an educational presentation about sustainable development"
                    onClick={(text) => {
                      setMessage(text);
                      setTimeout(() => {
                        const form = document.querySelector('form');
                        if (form) form.dispatchEvent(new Event('submit', { cancelable: true }));
                      }, 100);
                    }}
                  />
                </div>
              </div>
            </div>
          ) : (
            messages.map((msg, index) => (
              <MessageBubble 
                key={index} 
                message={msg[1]} 
                isUser={msg[0] !== null} 
                userMessage={msg[0]} 
                timestamp={new Date(Date.now() - (messages.length - index) * 2000)} 
              />
            ))
          )}

          {/* Generation in progress indicator */}
          {generatingPresentation && (
            <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-200 mb-4">
              <div className="flex space-x-3">
                <div className="flex-shrink-0">
                  <div className="h-10 w-10 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center">
                    <Presentation className="h-6 w-6 text-white" />
                  </div>
                </div>
                <div className="flex items-center flex-1">
                  <div className="text-gray-700">
                    <div className="flex items-center">
                      <p className="font-medium">Generating your presentation</p>
                      <div className="inline-flex ml-2">
                        <span className="animate-bounce mr-0.5 delay-0">.</span>
                        <span className="animate-bounce mr-0.5 delay-100">.</span>
                        <span className="animate-bounce mr-0.5 delay-200">.</span>
                      </div>
                    </div>
                    <div className="text-xs text-gray-500 flex items-center mt-1">
                      <Clock size={12} className="mr-1" />
                      <span>Time elapsed: {formatTime(elapsedTime)}</span>
                      <span className="ml-2 px-1.5 py-0.5 bg-blue-50 text-blue-700 rounded text-xs flex items-center">
                        <Zap size={10} className="mr-1" />
                        Processing
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Download button - improved positioning in chat area */}
          {downloadUrl && (
            <div className="mx-auto max-w-4xl px-3 sm:px-4 my-4">
              <div className="bg-white border border-blue-100 rounded-lg shadow-sm p-4">
                <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
                  <div className="text-center sm:text-left">
                    <p className="text-sm font-medium text-gray-700">Your presentation is ready!</p>
                    <p className="text-xs text-gray-500">{format(new Date(), 'PPpp')}</p>
                  </div>
                  
                  <button
                    onClick={handleDownload}
                    disabled={generatingPresentation || downloadClicked}
                    className={`w-full sm:w-auto ${downloadClicked ? 'bg-green-600' : 'bg-blue-600 hover:bg-blue-700'} text-white px-6 py-2 rounded-md font-medium transition flex items-center justify-center gap-2 shadow-sm disabled:opacity-50`}
                  >
                    <Presentation size={18} />
                    {downloadClicked ? 'Downloaded Successfully!' : 'Download Presentation'}
                  </button>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input area with improved mobile experience */}
      <div className="sticky bottom-0 left-0 right-0 px-3 sm:px-4 py-3 bg-white border-t z-30 flex justify-center">
        <form 
          onSubmit={handleSubmit} 
          className="w-full max-w-4xl"
        >
          <div className="flex rounded-md shadow-sm">
            <input
              id="message-input"
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              disabled={generatingPresentation}
              placeholder="Describe your presentation topic..."
              className="flex-1 p-3 border-none focus:ring-0 focus:outline-none text-base rounded-l-md input-compact"
            />
            <button
              type="submit"
              disabled={!message.trim() || generatingPresentation}
              className="bg-blue-600 text-white px-3 sm:px-4 py-2 sm:py-3 rounded-r-md hover:bg-blue-700 transition disabled:opacity-50 disabled:bg-blue-400 flex items-center"
            >
              {generatingPresentation ? (
                <div className="w-4 h-4 border-t-2 border-white rounded-full animate-spin"></div>
              ) : (
                <>
                  <span className="hidden sm:inline mr-2">Send</span>
                  <SendHorizontal size={18} />
                </>
              )}
            </button>
          </div>
          
          {/* Character count indicator */}
          <div className="mt-1 text-right text-xs text-gray-500">
            {message.length > 0 && `${message.length} characters`}
          </div>
        </form>
      </div>
    </div>
  );
};

// Component for suggestion buttons
interface SuggestionButtonProps {
  text: string;
  onClick: (text: string) => void;
}

const SuggestionButton: React.FC<SuggestionButtonProps> = ({ text, onClick }) => (
  <button
    className="w-full px-4 py-2 bg-blue-50 hover:bg-blue-100 text-blue-700 text-sm text-left rounded-md transition-colors border border-blue-100"
    onClick={() => onClick(text)}
  >
    "{text}"
  </button>
);

export default ChatInterface;