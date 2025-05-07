import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { useApi } from '../context/ApiContext';
import { FileText, Upload, CheckCircle, AlertCircle } from 'lucide-react';

export const PdfUploader: React.FC = () => {
  const { uploadPdf, uploadingPdf, pdfUploaded } = useApi();
  const [uploadError, setUploadError] = useState<string | null>(null);
  
  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (file) {
        if (file.type !== 'application/pdf') {
          setUploadError('Only PDF files are accepted');
          return;
        }
        
        try {
          setUploadError(null);
          await uploadPdf(file);
        } catch (error) {
          setUploadError('Failed to upload PDF');
          console.error('Upload error:', error);
        }
      }
    },
    [uploadPdf]
  );
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    multiple: false,
    disabled: uploadingPdf
  });
  
  return (
    <div>
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-md p-4 text-center cursor-pointer transition-colors ${
          isDragActive 
            ? 'border-blue-500 bg-blue-50' 
            : pdfUploaded 
              ? 'border-green-500 bg-green-50' 
              : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50'
        } ${uploadingPdf ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <input {...getInputProps()} />
        
        {pdfUploaded ? (
          <div className="flex flex-col items-center text-green-600">
            <CheckCircle size={24} className="mb-2" />
            <p className="text-sm">PDF uploaded successfully</p>
          </div>
        ) : uploadingPdf ? (
          <div className="flex flex-col items-center text-blue-600">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mb-2"></div>
            <p className="text-sm">Uploading PDF...</p>
          </div>
        ) : (
          <div className="flex flex-col items-center text-gray-500">
            <FileText size={24} className="mb-2" />
            <p className="text-sm font-medium">
              {isDragActive ? 'Drop the PDF here' : 'Drag & drop a PDF here'}
            </p>
            <p className="text-xs mt-1">or click to select</p>
          </div>
        )}
      </div>
      
      {uploadError && (
        <div className="mt-2 text-red-500 text-sm flex items-center">
          <AlertCircle size={16} className="mr-1" />
          {uploadError}
        </div>
      )}
    </div>
  );
};