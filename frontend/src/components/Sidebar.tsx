import React, { useState, useRef, useEffect } from 'react';
import { useApi } from '../context/ApiContext';
import { useAuth } from '../context/AuthContext';
import { 
  Layout as LayoutIcon, 
  Cpu, 
  Upload, 
  Download,
  FileText,
  ToggleLeft,
  ToggleRight,
  Trash2,
  LogOut,
  Sliders,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { TemplateSelector } from './TemplateSelector';
import { SlideCountSelector } from './SlideCountSelector';
import { OpenAIModelSelector } from './OpenAIModelSelector';
import { DocumentUploader } from './DocumentUploader';

interface SidebarProps {
  isInSettingsPanel?: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({ isInSettingsPanel = false }) => {
  const { 
    templates,
    currentTemplate,
    llmModel,
    useGDrive,
    usePdfContext,
    downloadUrl,
    clearingHistory,
    setTemplate,
    setLlmModel,
    setUseGDrive,
    setUsePdfContext,
    clearHistory,
    uploadToGDrive,
    changingModel,
    changingGDriveSettings,
    uploadingToDrive
  } = useApi();
  
  const { user, logout } = useAuth();

  const [model, setModel] = useState(llmModel);
  const [modelSource, setModelSource] = useState<'ollama' | 'openai'>('ollama');
  
  // Scroll position preservation
  const containerRef = useRef<HTMLDivElement>(null);
  const scrollPositionRef = useRef(0);
  
  // State for collapsible sections on mobile
  const [expandedSections, setExpandedSections] = useState({
    template: true,
    slideCount: true,
    model: isInSettingsPanel || window.innerWidth > 768,
    gdrive: isInSettingsPanel || window.innerWidth > 768,
    pdf: true,
    document: true
  });

  // Save scroll position before state changes
  const saveScrollPosition = () => {
    if (isInSettingsPanel && containerRef.current) {
      scrollPositionRef.current = containerRef.current.scrollTop;
    }
  };

  // Restore scroll position after component updates
  useEffect(() => {
    if (isInSettingsPanel && containerRef.current) {
      // Use requestAnimationFrame to ensure the DOM has fully updated before restoring scroll
      const restoreScroll = () => {
        if (containerRef.current) {
          containerRef.current.scrollTop = scrollPositionRef.current;
        }
      };
      
      // Queue scroll restoration after render is complete
      const timeoutId = setTimeout(() => {
        requestAnimationFrame(restoreScroll);
      }, 0);
      
      return () => clearTimeout(timeoutId);
    }
  }, [
    isInSettingsPanel,
    expandedSections,
    model,
    modelSource,
    useGDrive,
    usePdfContext,
    downloadUrl,
    clearingHistory,
    changingModel,
    changingGDriveSettings,
    uploadingToDrive
  ]);

  const toggleSection = (section: keyof typeof expandedSections) => {
    saveScrollPosition();
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };
  
  const handleModelChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    saveScrollPosition();
    setModel(e.target.value);
  };
  
  const handleModelSubmit = async () => {
    saveScrollPosition();
    if (model.trim()) {
      await setLlmModel(model);
    }
  };
  
  const handleGDriveUpload = async () => {
    saveScrollPosition();
    if (!downloadUrl) return;
    
    const result = await uploadToGDrive();
    if (result.success && result.viewLink) {
      window.open(result.viewLink, '_blank');
    }
  };
  
  // Collapsible section component
  const CollapsibleSection = ({ 
    title, 
    icon, 
    sectionKey,
    children 
  }: { 
    title: string; 
    icon: React.ReactNode;
    sectionKey: keyof typeof expandedSections;
    children: React.ReactNode;
  }) => (
    <div className="space-y-3 pb-4 border-b">
      <div 
        className="flex items-center justify-between cursor-pointer"
        onClick={() => toggleSection(sectionKey)}
      >
        <div className="flex items-center text-gray-700">
          {icon}
          <h3 className="font-medium">{title}</h3>
        </div>
        {expandedSections[sectionKey] ? (
          <ChevronUp size={16} className="text-gray-500" />
        ) : (
          <ChevronDown size={16} className="text-gray-500" />
        )}
      </div>
      
      {expandedSections[sectionKey] && (
        <div className="transition-all duration-300 ease-in-out">
          {children}
        </div>
      )}
    </div>
  );
  
  // Apply conditional classes based on where the sidebar is being used
  const containerClasses = isInSettingsPanel 
    ? "p-3 sm:p-4 space-y-5 overflow-y-auto overflow-x-hidden max-w-3xl mx-auto h-full"
    : "p-3 sm:p-4 space-y-5 overflow-y-auto overflow-x-hidden";

  return (
    <div 
      className={containerClasses} 
      ref={containerRef}
      key={`sidebar-container-${isInSettingsPanel ? 'settings' : 'sidebar'}`}
    >
      {/* Only show user profile when not in settings panel (it's redundant there) */}
      {!isInSettingsPanel && user && (
        <div className="flex items-center space-x-2 sm:space-x-3 p-2 sm:p-3 bg-gray-50 rounded-lg mb-4">
          <img
            src={user.picture}
            alt={user.name}
            className="w-8 h-8 sm:w-10 sm:h-10 rounded-full"
          />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-gray-700 truncate">{user.name}</p>
            <p className="text-xs text-gray-500 truncate">{user.email}</p>
          </div>
          <button 
            onClick={logout}
            className="text-gray-500 hover:text-red-500"
            title="Sign out"
          >
            <LogOut size={16} className="sm:hidden" />
            <LogOut size={18} className="hidden sm:block" />
          </button>
        </div>
      )}

      {/* Template Selection */}
      <CollapsibleSection 
        title="Select Template" 
        icon={<LayoutIcon size={18} className="mr-2" />}
        sectionKey="template"
      >
        <TemplateSelector
          templates={templates}
          currentTemplate={currentTemplate}
          onChange={(template) => {
            saveScrollPosition();
            setTemplate(template);
          }}
        />
      </CollapsibleSection>
      
      {/* Slide Count Selection */}
      <CollapsibleSection 
        title="Slide Count" 
        icon={<Sliders size={18} className="mr-2" />}
        sectionKey="slideCount"
      >
        <SlideCountSelector />
      </CollapsibleSection>
      
      {/* Model Selection */}
      <CollapsibleSection 
        title="AI Model" 
        icon={<Cpu size={18} className="mr-2" />}
        sectionKey="model"
      >
        <div className="space-y-2">
          <div className="flex space-x-2">
            <button
              className={`flex-1 px-3 py-1.5 text-sm rounded-md ${
                modelSource === 'ollama'
                  ? 'bg-blue-100 text-blue-800 font-medium'
                  : 'bg-gray-100 text-gray-700'
              }`}
              onClick={() => {
                saveScrollPosition();
                setModelSource('ollama');
              }}
            >
              Ollama
            </button>
            <button
              className={`flex-1 px-3 py-1.5 text-sm rounded-md ${
                modelSource === 'openai'
                  ? 'bg-blue-100 text-blue-800 font-medium'
                  : 'bg-gray-100 text-gray-700'
              }`}
              onClick={() => {
                saveScrollPosition();
                setModelSource('openai');
              }}
            >
              OpenAI
            </button>
          </div>
        </div>
        
        {modelSource === 'ollama' ? (
          <div className="space-y-2">
            <div className="flex mt-2">
              <input
                type="text"
                value={model}
                onChange={handleModelChange}
                placeholder="Model name (e.g. llama3)"
                className="flex-1 px-3 py-2 border rounded-l-md focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
              />
              <button
                onClick={handleModelSubmit}
                className="bg-blue-600 text-white px-3 py-2 rounded-r-md hover:bg-blue-700 transition"
                disabled={changingModel}
              >
                {changingModel ? (
                  <div className="w-4 h-4 border-t-2 border-white rounded-full animate-spin mx-auto"></div>
                ) : (
                  'Set'
                )}
              </button>
            </div>
            <p className="text-xs text-gray-500">Enter the name of any Ollama model you have installed locally.</p>
          </div>
        ) : (
          <OpenAIModelSelector />
        )}
      </CollapsibleSection>
      
      {/* Google Drive Toggle */}
      <CollapsibleSection 
        title="Upload to Google Drive" 
        icon={<Download size={18} className="mr-2" />}
        sectionKey="gdrive"
      >
        <div className="flex items-center justify-between">
          <p className="text-sm text-gray-700">Auto Upload</p>
          <button
            onClick={() => {
              saveScrollPosition();
              setUseGDrive(!useGDrive);
            }}
            className="focus:outline-none"
            disabled={changingGDriveSettings}
            aria-label={useGDrive ? "Disable Google Drive upload" : "Enable Google Drive upload"}
          >
            {changingGDriveSettings ? (
              <div className="w-4 h-4 border-t-2 border-blue-500 rounded-full animate-spin"></div>
            ) : useGDrive ? (
              <ToggleRight className="h-6 w-6 text-blue-600" />
            ) : (
              <ToggleLeft className="h-6 w-6 text-gray-400" />
            )}
          </button>
        </div>
        
        {downloadUrl && (
          <button
            onClick={handleGDriveUpload}
            disabled={uploadingToDrive}
            className="w-full mt-3 flex items-center justify-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50"
          >
            {uploadingToDrive ? (
              <div className="w-4 h-4 border-t-2 border-white rounded-full animate-spin"></div>
            ) : (
              <>
                <Upload size={16} className="mr-2" />
                Upload to Drive
              </>
            )}
          </button>
        )}
      </CollapsibleSection>
      
      {/* Document Upload with RAG */}
      <CollapsibleSection 
        title="Document Upload for RAG" 
        icon={<FileText size={18} className="mr-2" />}
        sectionKey="document"
      >
        <DocumentUploader />
        <div className="flex items-center justify-between mt-3">
          <p className="text-sm text-gray-700">Use Document Context</p>
          <button
            onClick={() => {
              saveScrollPosition();
              setUsePdfContext(!usePdfContext);
            }}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-600 focus:ring-offset-2 ${
              usePdfContext ? 'bg-blue-600' : 'bg-gray-200'
            }`}
          >
            <span
              className={`${
                usePdfContext ? 'translate-x-6' : 'translate-x-1'
              } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
            />
          </button>
        </div>
      </CollapsibleSection>
      
      {/* Clear Chat History Button - more visible and clear intent */}
      <div className="pt-2">
        <button
          onClick={() => {
            saveScrollPosition();
            clearHistory();
          }}
          disabled={clearingHistory}
          className="w-full flex items-center justify-center px-4 py-2 border border-red-300 text-red-600 rounded-md hover:bg-red-50 transition-colors disabled:opacity-50"
        >
          {clearingHistory ? (
            <div className="w-4 h-4 border-t-2 border-red-500 rounded-full animate-spin"></div>
          ) : (
            <>
              <Trash2 size={16} className="mr-2" />
              Clear Chat History
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default Sidebar;