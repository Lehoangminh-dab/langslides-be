import React, { useState, useEffect } from 'react';
import { useApi } from '../context/ApiContext';
import { KeyRound, Eye, EyeOff, CheckCircle } from 'lucide-react';

export const OpenAIModelSelector: React.FC = () => {
  const { setLlmModel, changingModel, llmModel } = useApi();
  const [apiKey, setApiKey] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [selectedModel, setSelectedModel] = useState('gpt-4o');
  const [keyStored, setKeyStored] = useState(false);

  // Check if the current LLM model is an OpenAI one
  useEffect(() => {
    // If current model is an OpenAI model, it means a key is stored
    setKeyStored(llmModel && (
      llmModel.includes('gpt-4') || 
      llmModel.includes('gpt-3.5') ||
      llmModel.includes('text-davinci')
    ));
  }, [llmModel]);

  const handleSubmit = async () => {
    if (apiKey.trim()) {
      await setLlmModel(selectedModel, 'openai', apiKey);
      setKeyStored(true);
      setApiKey(''); // Clear input field after submission
    }
  };

  return (
    <div className="space-y-3">
      <h3 className="font-medium text-sm text-gray-700">OpenAI Model</h3>
      
      {keyStored && (
        <div className="flex items-center space-x-2 text-green-600 text-sm mb-2">
          <CheckCircle size={16} />
          <span>API key stored in your session</span>
        </div>
      )}
      
      <div>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={changingModel}
        >
          <option value="gpt-4o">GPT-4o</option>
          <option value="gpt-4-turbo">GPT-4 Turbo</option>
          <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
        </select>
      </div>
      <div className="relative">
        <input
          type={showApiKey ? "text" : "password"}
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="Enter your OpenAI API key"
          className="w-full px-3 py-2 pl-8 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <KeyRound size={16} className="absolute left-2 top-2.5 text-gray-400" />
        <button
          type="button"
          onClick={() => setShowApiKey(!showApiKey)}
          className="absolute right-2 top-2.5 text-gray-400"
        >
          {showApiKey ? <EyeOff size={16} /> : <Eye size={16} />}
        </button>
      </div>
      <button
        onClick={handleSubmit}
        disabled={!apiKey.trim() || changingModel}
        className="w-full bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 transition disabled:opacity-50 flex items-center justify-center gap-2"
      >
        {changingModel ? (
          <div className="w-4 h-4 border-t-2 border-white rounded-full animate-spin"></div>
        ) : (
          'Use OpenAI'
        )}
      </button>
      <p className="text-xs text-gray-500">
        Your API key is stored in your session and will be forgotten when you log out.
      </p>
    </div>
  );
};