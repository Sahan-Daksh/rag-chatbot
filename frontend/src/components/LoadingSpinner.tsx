import { Loader } from 'lucide-react';

export const LoadingSpinner = () => (
  <div className="flex justify-center">
    <Loader className="w-6 h-6 animate-spin text-blue-500" />
  </div>
);