import React, { useState } from 'react';
import { Check } from 'lucide-react';

interface TemplateSelectorProps {
  templates: string[];
  currentTemplate: string;
  onChange: (template: string) => Promise<void>;
}

export const TemplateSelector: React.FC<TemplateSelectorProps> = ({
  templates,
  currentTemplate,
  onChange
}) => {
  const [changingTemplate, setChangingTemplate] = useState(false);

  const setTemplate = async (template: string) => {
    if (changingTemplate || currentTemplate === template) return;
    setChangingTemplate(true);
    await onChange(template);
    setChangingTemplate(false);
  };

  if (templates.length === 0) {
    return <div className="text-gray-500 text-sm">Loading templates...</div>;
  }
  
  return (
    <div className="grid grid-cols-2 gap-2">
      {templates.map((template) => (
        <button
          key={template}
          onClick={() => setTemplate(template)}
          disabled={changingTemplate || currentTemplate === template}
          className={`p-3 rounded-md border transition-all ${
            currentTemplate === template
              ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200'
              : 'border-gray-200 hover:border-blue-300 hover:bg-blue-50'
          } ${changingTemplate ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {changingTemplate && currentTemplate === template ? (
            <div className="w-4 h-4 border-t-2 border-white rounded-full animate-spin mx-auto"></div>
          ) : (
            <>
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium">{template}</span>
                {currentTemplate === template && (
                  <Check size={16} className="text-blue-600" />
                )}
              </div>
              <div className="h-12 bg-gray-100 rounded-sm flex items-center justify-center">
                <span className="text-xs text-gray-500">Template Preview</span>
              </div>
            </>
          )}
        </button>
      ))}
    </div>
  );
};