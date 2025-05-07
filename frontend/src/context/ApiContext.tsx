import React, { createContext, ReactNode, useEffect, useState, useContext } from 'react';
import axios from 'axios';
import { useAuth } from './AuthContext';

// API base URL - adjust as needed for your environment
const API_BASE_URL = 'http://localhost:8000'; // Replace with your actual API base URL
// Type definitions
export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface UserProfile {
  id: number;
  email: string;
  name: string;
  picture_url: string;
  created_at: string;
  last_login: string;
}

export interface PdfUpload {
  id: number;
  file_name: string;
  collection_name: string;
  created_at: string;
  file_hash: string;
}

export interface Presentation {
  id: number;
  session_id: string;
  file_path: string;
  template: string;
  slide_count: number;
  created_at: string;
  gdrive_link: string | null;
  download_count: number;
}

export interface Session {
  id: string;
  created_at: string;
  last_active: string;
  message_count: number;
}

export interface ApiContextType {
  sessionId: string | null;
  initialLoading: boolean;
  generatingPresentation: boolean;
  setGeneratingPresentation: React.Dispatch<React.SetStateAction<boolean>>;
  uploadingPdf: boolean;
  changingTemplate: boolean;
  changingModel: boolean;
  changingGDriveSettings: boolean;
  uploadingToDrive: boolean;
  clearingHistory: boolean;
  templates: string[];
  currentTemplate: string;
  llmModel: string;
  useGDrive: boolean;
  usePdfContext: boolean;
  pdfUploaded: boolean;
  slideCount: number;
  messages: [string | null, string | null][];
  setMessages: React.Dispatch<React.SetStateAction<[string | null, string | null][]>>;
  downloadUrl: string | null;
  apiConnected: boolean;
  generatePresentation: (message: string, isNewPresentation?: boolean) => Promise<void>;
  setTemplate: (template: string) => Promise<void>;
  setLlmModel: (model: string, provider?: string, apiKey?: string) => Promise<void>;
  setUseGDrive: (enable: boolean) => Promise<void>;
  setUsePdfContext: (enable: boolean) => Promise<void>;
  setSlideCount: (count: number) => void;
  uploadPdf: (file: File) => Promise<void>;
  uploadDocument: (file: File) => Promise<void>;
  clearHistory: () => Promise<void>;
  uploadToGDrive: () => Promise<{ success: boolean; viewLink?: string }>;
  downloadPresentation: () => void;

  // New database related properties
  userProfile: UserProfile | null;
  pdfUploads: PdfUpload[];
  presentations: Presentation[];
  sessions: Session[];
  loadingUserData: boolean;

  // New functions
  fetchUserData: () => Promise<void>;
}

// Create context with default values
export const ApiContext = createContext<ApiContextType>({
  sessionId: null,
  initialLoading: true,
  generatingPresentation: false,
  setGeneratingPresentation: () => {},
  uploadingPdf: false,
  changingTemplate: false,
  changingModel: false,
  changingGDriveSettings: false,
  uploadingToDrive: false,
  clearingHistory: false,
  templates: [],
  currentTemplate: 'Basic',
  llmModel: '',
  useGDrive: false,
  usePdfContext: false,
  pdfUploaded: false,
  slideCount: 10,
  messages: [],
  setMessages: () => {},
  downloadUrl: null,
  apiConnected: false,
  generatePresentation: async () => {},
  setTemplate: async () => {},
  setLlmModel: async () => {},
  setUseGDrive: async () => {},
  setUsePdfContext: async () => {},
  setSlideCount: () => {},
  uploadPdf: async () => {},
  uploadDocument: async () => {},
  clearHistory: async () => {},
  uploadToGDrive: async () => ({ success: false }),
  downloadPresentation: () => {},

  // New database related properties
  userProfile: null,
  pdfUploads: [],
  presentations: [],
  sessions: [],
  loadingUserData: false,

  // New functions
  fetchUserData: async () => {},
});

interface ApiProviderProps {
  children: ReactNode;
}

const ApiProvider: React.FC<ApiProviderProps> = ({ children }) => {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [initialLoading, setInitialLoading] = useState<boolean>(true);
  const [generatingPresentation, setGeneratingPresentation] = useState<boolean>(false);
  const [uploadingPdf, setUploadingPdf] = useState<boolean>(false);
  const [changingTemplate, setChangingTemplate] = useState<boolean>(false);
  const [changingModel, setChangingModel] = useState<boolean>(false);
  const [changingGDriveSettings, setChangingGDriveSettings] = useState<boolean>(false);
  const [uploadingToDrive, setUploadingToDrive] = useState<boolean>(false);
  const [clearingHistory, setClearingHistory] = useState<boolean>(false);
  const [templates, setTemplates] = useState<string[]>([]);
  const [currentTemplate, setCurrentTemplate] = useState<string>('Basic');
  const [llmModel, setLlmModelState] = useState<string>('mistral:v0.2');
  const [useGDrive, setUseGDriveState] = useState<boolean>(false);
  const [usePdfContext, setUsePdfContextState] = useState<boolean>(false);
  const [pdfUploaded, setPdfUploaded] = useState<boolean>(false);
  const [slideCount, setSlideCount] = useState<number>(10);
  const [messages, setMessages] = useState<[string | null, string | null][]>([]);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [apiConnected, setApiConnected] = useState<boolean>(false);

  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [pdfUploads, setPdfUploads] = useState<PdfUpload[]>([]);
  const [presentations, setPresentations] = useState<Presentation[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loadingUserData, setLoadingUserData] = useState<boolean>(false);

  const { isAuthenticated, checkAuth } = useAuth();

  // Add this function to handle session creation with session ID persistence
  const createSession = async () => {
    try {
      // Try to get existing session ID from localStorage
      const existingSessionId = localStorage.getItem('sessionId');
      
      const { data } = await axios.post(
        `${API_BASE_URL}/api/session`,
        existingSessionId ? { session_id: existingSessionId } : {},
        { withCredentials: true }
      );
      
      // Store the session ID in localStorage
      localStorage.setItem('sessionId', data.session_id);
      setSessionId(data.session_id);
      
      // Initialize with greeting message if any
      if (data.greeting) {
        setMessages([[null, data.greeting]]);
      }
      
      return data.session_id;
    } catch (error) {
      console.error('Error creating session:', error);
      return null;
    }
  };

  // Initialize session and load templates
  useEffect(() => {
    const initSession = async () => {
      try {
        setInitialLoading(true);

        // Add health check to verify API connection
        try {
          // Try to connect to API health endpoint
          await axios.get(`${API_BASE_URL}/health`, {
            timeout: 5000, // 5-second timeout
          });
          setApiConnected(true);
        } catch (error) {
          console.error('API server not available. Is the backend running?');
          setApiConnected(false);
          setInitialLoading(false);
          return; // Exit early
        }

        // Continue with session initialization if API is connected
        const sessionId = await createSession();
        if (!sessionId) {
          setInitialLoading(false);
          return;
        }

        // Load templates
        const templatesResponse = await axios.get(`${API_BASE_URL}/api/templates`, { withCredentials: true });
        setTemplates(templatesResponse.data.templates || []);
        setCurrentTemplate(templatesResponse.data.current_template || 'Basic');

        setInitialLoading(false);
      } catch (error) {
        console.error('Error initializing session:', error);
        if (axios.isAxiosError(error) && error.code === 'ERR_NETWORK') {
          setApiConnected(false);
          alert('Cannot connect to the API server. Please make sure the backend is running.');
        }
        setInitialLoading(false);
      }
    };

    initSession();
  }, []);

  // Generate presentation
  const generatePresentation = async (message: string) => {
    if (!sessionId) return;

    try {
      setGeneratingPresentation(true);

      const { data } = await axios.post(
        `${API_BASE_URL}/api/generate`,
        {
          session_id: sessionId,
          message: message,
          use_pdf_context: usePdfContext,
          slide_count: slideCount,
          is_new_presentation: !downloadUrl // Mark as new presentation if no current one
        },
        { withCredentials: true }
      );

      if (data.success) {
        if (data.history) {
          setMessages(data.history);
        }
        
        if (data.download_url) {
          setDownloadUrl(data.download_url);
        }
      }
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.status === 401) {
        await checkAuth();
      }
      console.error('Error generating presentation:', error);
      
      // Add error message to chat
      setMessages(prev => {
        // Find the last user message
        const lastUserMsgIndex = [...prev].reverse().findIndex(m => m[0] !== null);
        if (lastUserMsgIndex !== -1) {
          const newMessages = [...prev];
          const actualIndex = prev.length - 1 - lastUserMsgIndex;
          newMessages[actualIndex] = [
            newMessages[actualIndex][0], 
            "Sorry, there was an error generating your presentation. Please try again."
          ];
          return newMessages;
        }
        return prev;
      });
    } finally {
      setGeneratingPresentation(false);
    }
  };

  // Set template
  const setTemplate = async (template: string) => {
    if (!sessionId) return;

    try {
      setChangingTemplate(true);
      const { data } = await axios.post(
        `${API_BASE_URL}/api/set-template`,
        {
          template_name: template,
        },
        { withCredentials: true }
      );

      if (data.success) {
        setCurrentTemplate(template);
      }
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.status === 401) {
        await checkAuth();
      }
      console.error('Error setting template:', error);
    } finally {
      setChangingTemplate(false);
    }
  };

  // Set LLM model
  const setLlmModel = async (model: string, provider?: string, apiKey?: string) => {
    if (!model.trim() || !sessionId) return;

    try {
      setChangingModel(true);
      
      const { data } = await axios.post(
        `${API_BASE_URL}/api/set-llm`,
        {
          model_name: model,
          provider: provider,
          api_key: apiKey
        },
        { withCredentials: true }
      );

      if (data.success) {
        setLlmModelState(model);
      }
    } catch (error) {
      console.error('Error setting LLM model:', error);
    } finally {
      setChangingModel(false);
    }
  };

  // Set Google Drive upload setting
  const setUseGDrive = async (enable: boolean) => {
    if (!sessionId) return;

    try {
      setChangingGDriveSettings(true);
      const { data } = await axios.post(
        `${API_BASE_URL}/api/set-gdrive-upload`,
        {
          enable,
        },
        { withCredentials: true }
      );

      if (data.success) {
        setUseGDriveState(enable);
      }
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.status === 401) {
        await checkAuth();
      }
      console.error('Error setting Google Drive upload:', error);
    } finally {
      setChangingGDriveSettings(false);
    }
  };

  // Toggle PDF context usage
  const setUsePdfContext = (enable: boolean) => {
    setUsePdfContextState(enable);
  };

  // Update the uploadPdf function to send session ID
  const uploadPdf = async (file: File) => {
    if (!file || !sessionId) return;

    try {
      setUploadingPdf(true);
      
      const formData = new FormData();
      formData.append('file', file);
      formData.append('session_id', sessionId);
      
      const { data } = await axios.post(
        `${API_BASE_URL}/api/process-pdf`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          withCredentials: true
        }
      );
      
      setPdfUploaded(true);
      return data;
    } catch (error) {
      console.error('Error uploading PDF:', error);
      return { success: false, error: 'Failed to upload PDF' };
    } finally {
      setUploadingPdf(false);
    }
  };

  // Add a new function to handle document uploads
  const uploadDocument = async (file: File) => {
    if (!file || !sessionId) return;

    try {
      setUploadingPdf(true); // Reusing the PDF loading state for now
      
      const formData = new FormData();
      formData.append('file', file);
      formData.append('session_id', sessionId);
      
      const { data } = await axios.post(
        `${API_BASE_URL}/api/process-document`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          withCredentials: true
        }
      );
      
      setPdfUploaded(true); // Reuse the PDF uploaded state
      return data;
    } catch (error) {
      console.error('Error uploading document:', error);
      return { success: false, error: 'Failed to upload document' };
    } finally {
      setUploadingPdf(false);
    }
  };

  // Clear chat history
  const clearHistory = async () => {
    if (!sessionId) return;

    try {
      setClearingHistory(true);
      const { data } = await axios.post(
        `${API_BASE_URL}/api/clear-history`,
        {
          session_id: sessionId,
        },
        { withCredentials: true }
      );

      if (data.success) {
        // Reset messages with just the greeting
        setMessages(data.greeting ? [[null, data.greeting]] : []);
        setDownloadUrl(null);
      }
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.status === 401) {
        await checkAuth();
      }
      console.error('Error clearing history:', error);
    } finally {
      setClearingHistory(false);
    }
  };

  // Upload to Google Drive
  const uploadToGDrive = async () => {
    if (!sessionId) return { success: false };

    try {
      setUploadingToDrive(true);
      const { data } = await axios.post(
        `${API_BASE_URL}/api/upload-to-gdrive`,
        {
          session_id: sessionId,
        },
        { withCredentials: true }
      );

      if (data.success) {
        return { success: true, viewLink: data.view_link };
      } else {
        console.error('Google Drive upload failed:', data.message);
        return { success: false };
      }
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.status === 401) {
        await checkAuth();
      }
      console.error('Error uploading to Google Drive:', error);
      return { success: false };
    } finally {
      setUploadingToDrive(false);
    }
  };

  // Download presentation
  const downloadPresentation = () => {
    if (downloadUrl && sessionId) {
      window.open(`${API_BASE_URL}${downloadUrl}`, '_blank');
    }
  };

  // Fetch user data
  const fetchUserData = async () => {
    if (!isAuthenticated) return;

    // Prevent multiple simultaneous requests
    if (loadingUserData) return;

    setLoadingUserData(true);
    try {
      // Use a single Promise.all to fetch all data at once
      const [profileRes, uploadsRes, presentationsRes, sessionsRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/api/user-profile`, { withCredentials: true }),
        axios.get(`${API_BASE_URL}/api/user-uploads`, { withCredentials: true }),
        axios.get(`${API_BASE_URL}/api/user-presentations`, { withCredentials: true }),
        axios.get(`${API_BASE_URL}/api/user-sessions`, { withCredentials: true }),
      ]);

      // Update state with all results
      if (profileRes.data.success) {
        setUserProfile(profileRes.data.profile);
      }

      if (uploadsRes.data.success) {
        setPdfUploads(uploadsRes.data.uploads);
      }

      if (presentationsRes.data.success) {
        setPresentations(presentationsRes.data.presentations);
      }

      if (sessionsRes.data.success) {
        setSessions(sessionsRes.data.sessions);
      }
    } catch (error) {
      console.error('Error fetching user data:', error);
    } finally {
      setLoadingUserData(false);
    }
  };

  const value = {
    sessionId,
    initialLoading,
    generatingPresentation,
    setGeneratingPresentation,
    uploadingPdf,
    changingTemplate,
    changingModel,
    changingGDriveSettings,
    uploadingToDrive,
    clearingHistory,
    templates,
    currentTemplate,
    llmModel,
    useGDrive,
    usePdfContext,
    pdfUploaded,
    slideCount,
    messages,
    setMessages,
    downloadUrl,
    apiConnected,
    generatePresentation,
    setTemplate,
    setLlmModel,
    setUseGDrive,
    setUsePdfContext,
    setSlideCount,
    uploadPdf,
    uploadDocument,
    clearHistory,
    uploadToGDrive,
    downloadPresentation,
    userProfile,
    pdfUploads,
    presentations,
    sessions,
    loadingUserData,
    fetchUserData,
  };

  return <ApiContext.Provider value={value}>{children}</ApiContext.Provider>;
};

// Add this custom hook to use the API context
export const useApi = () => {
  const context = useContext(ApiContext);
  if (context === undefined) {
    throw new Error('useApi must be used within an ApiProvider');
  }
  return context;
};

export { ApiProvider };