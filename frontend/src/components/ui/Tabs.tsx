import React, { createContext, useContext, useState } from 'react';

interface TabsContextType {
  value: string;
  onChange: (value: string) => void;
}

const TabsContext = createContext<TabsContextType | undefined>(undefined);

export const Tabs: React.FC<{
  defaultValue: string;
  value?: string;
  onValueChange?: (value: string) => void;
  className?: string;
  children: React.ReactNode;
}> = ({ defaultValue, value, onValueChange, className = '', children }) => {
  const [tabValue, setTabValue] = useState(value || defaultValue);
  
  const handleValueChange = (newValue: string) => {
    setTabValue(newValue);
    onValueChange?.(newValue);
  };
  
  return (
    <TabsContext.Provider value={{ value: tabValue, onChange: handleValueChange }}>
      <div className={className}>
        {children}
      </div>
    </TabsContext.Provider>
  );
};

export const TabsList: React.FC<{
  className?: string;
  children: React.ReactNode;
}> = ({ className = '', children }) => {
  return (
    <div className={className}>
      {children}
    </div>
  );
};

export const TabsTrigger: React.FC<{
  value: string;
  disabled?: boolean;
  className?: string;
  activeClassName?: string;
  inactiveClassName?: string;
  children: React.ReactNode;
}> = ({ 
  value, 
  disabled, 
  className = '', 
  activeClassName = '', 
  inactiveClassName = '',
  children 
}) => {
  const context = useContext(TabsContext);
  if (!context) throw new Error('TabsTrigger must be used within Tabs');
  
  const isActive = context.value === value;
  const combinedClassName = `${className} ${isActive ? activeClassName : inactiveClassName}`;
  
  return (
    <button
      type="button"
      role="tab"
      aria-selected={isActive}
      disabled={disabled}
      className={combinedClassName}
      onClick={() => context.onChange(value)}
    >
      {children}
    </button>
  );
};

export const TabsContent: React.FC<{
  value: string;
  className?: string;
  children: React.ReactNode;
}> = ({ value, className = '', children }) => {
  const context = useContext(TabsContext);
  if (!context) throw new Error('TabsContent must be used within Tabs');
  
  if (context.value !== value) return null;
  
  return (
    <div role="tabpanel" className={className}>
      {children}
    </div>
  );
};