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