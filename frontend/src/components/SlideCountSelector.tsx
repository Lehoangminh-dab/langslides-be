import React from 'react';
import { Sliders } from 'lucide-react';
import { useApi } from '../context/ApiContext';

export const SlideCountSelector: React.FC = () => {
  const { slideCount, setSlideCount } = useApi();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const count = parseInt(e.target.value, 10);
    setSlideCount(count);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center text-gray-700">
        <Sliders size={18} className="mr-2" />
        <h3 className="font-medium">Slide Count: {slideCount}</h3>
      </div>
      <div className="flex items-center">
        <span className="text-xs text-gray-500 mr-2">3</span>
        <input
          type="range"
          min={3}
          max={20}
          value={slideCount}
          onChange={handleChange}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
        <span className="text-xs text-gray-500 ml-2">20</span>
      </div>
      <p className="text-xs text-gray-500">
        Select the number of slides to generate
      </p>
    </div>
  );
};