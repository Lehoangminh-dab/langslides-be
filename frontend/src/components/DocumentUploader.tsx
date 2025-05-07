import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { useApi } from '../context/ApiContext';
import { ChevronRight, File, FileText, Upload, AlertCircle, CheckCircle, Loader2 } from 'lucide-react';

interface DocumentUploaderProps {
  onSuccess?: (info: any) => void;
}

export const DocumentUploader: React.FC<DocumentUploaderProps> = ({ onSuccess }) => {
  const { uploadDocument, uploadingPdf, pdfUploaded } = useApi();
  const [uploadError, setUploadError] = useState<string | null>(null);
  
  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;
      
      // Check file type
      if (!file.type.includes('pdf') && 
          !file.type.includes('doc') && 
          !file.name.toLowerCase().endsWith('.doc') && 
          !file.name.toLowerCase().endsWith('.docx')) {
        setUploadError('Please upload a PDF, DOC, or DOCX file');
        return;
      }
      
      // Check file size (limit to 15MB)
      if (file.size > 15 * 1024 * 1024) {
        setUploadError('Please upload a document smaller than 15MB');
        return;
      }
      
      try {
        setUploadError(null);
        // Use the context's uploadDocument function instead
        const result = await uploadDocument(file);
        if (onSuccess) {
          onSuccess(result);
        }
      } catch (error) {
        setUploadError('Failed to upload document');
        console.error('Upload error:', error);
      }
    },
    [uploadDocument, onSuccess]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    disabled: uploadingPdf,
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    }
  });

  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-md p-4 text-center cursor-pointer transition-colors
          ${isDragActive 
            ? 'border-blue-500 bg-blue-50' 
            : pdfUploaded 
              ? 'border-green-500 bg-green-50' 
              : 'border-gray-300 hover:border-blue-400'}
          ${uploadingPdf ? 'opacity-50' : ''}
        `}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center justify-center text-sm text-gray-500">
          {pdfUploaded ? (
            <div className="flex flex-col items-center text-green-600">
              <CheckCircle size={24} className="mb-2" />
              <p className="text-sm">Document uploaded successfully</p>
            </div>
          ) : uploadingPdf ? (
            <div className="flex flex-col items-center">
              <Loader2 size={24} className="mb-2 animate-spin text-blue-500" />
              <p className="text-sm">Uploading document...</p>
            </div>
          ) : (
            <>
              <Upload className="h-10 w-10 text-gray-400 mb-2" />
              <p className="mb-1 font-medium">
                {isDragActive
                  ? "Drop the document here"
                  : "Drag & drop a document or click to select"}
              </p>
              <p className="text-xs">
                Supports PDF, DOC, and DOCX files (max 15MB)
              </p>
            </>
          )}
        </div>
      </div>
      
      {uploadError && (
        <div className="mt-2 text-red-500 text-sm flex items-center">
          <AlertCircle size={16} className="mr-1" />
          {uploadError}
        </div>
      )}
      
      <div className="flex flex-col space-y-1 mt-2 text-xs text-gray-500">
        <div className="flex items-center">
          <ChevronRight className="inline h-3 w-3 mr-1" />
          <FileText className="inline h-3 w-3 mr-1" />
          <span>PDF files will extract text and embedded images</span>
        </div>
        <div className="flex items-center">
          <ChevronRight className="inline h-3 w-3 mr-1" />
          <File className="inline h-3 w-3 mr-1" />
          <span>DOC/DOCX files will extract text, tables, and images</span>
        </div>
      </div>
    </div>
  );
};