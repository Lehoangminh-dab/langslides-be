import { useAuth } from '../context/AuthContext';
import { FcGoogle } from 'react-icons/fc';

const Login = () => {
  const { login, isLoading } = useAuth();

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-50">
      <div className="w-full max-w-md p-8 space-y-8 bg-white rounded-lg shadow-lg">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-800">Presentation Generator</h1>
          <p className="mt-2 text-gray-600">Sign in to create AI-powered presentations</p>
        </div>
        
        <button
          onClick={login}
          disabled={isLoading}
          className="flex items-center justify-center w-full px-4 py-3 space-x-3 text-gray-700 transition-colors duration-300 bg-white border border-gray-300 rounded-md shadow hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
        >
          <FcGoogle size={24} />
          <span>Sign in with Google</span>
        </button>
      </div>
    </div>
  );
};

export default Login;