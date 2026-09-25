import React, { useState } from 'react';

interface AuthModalProps {
  onLoginSuccess: (token: string, userData?: any) => void;
  onSwitchToOnboarding: (userData: { email: string; pass: string; name: string }) => void;
}

export const AuthModal: React.FC<AuthModalProps> = ({ onLoginSuccess, onSwitchToOnboarding }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isLogin) {
      onSwitchToOnboarding({ email, pass: password, name: fullName });
      return;
    }

    try {
      const res = await fetch('http://127.0.0.1:8000/api/v1/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      const responseData = await res.json();

      if (res.ok) {
        const token = responseData?.data?.access_token || responseData?.access_token;
        const user = responseData?.data || responseData?.user;

        localStorage.setItem('token', token);
        onLoginSuccess(token, user);
      } else {
        alert(responseData.detail || responseData.error || 'Login failed');
      }
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-slate-900 text-slate-100 p-4">
      <div className="w-full max-w-md bg-slate-800 border border-slate-700 rounded-2xl p-8 shadow-xl">
        <h2 className="text-2xl font-bold text-center mb-6">{isLogin ? 'Welcome Back to ELARA' : 'Create Account'}</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          {!isLogin && (
            <div>
              <label className="block text-sm mb-1">Full Name</label>
              <input type="text" value={fullName} onChange={(e) => setFullName(e.target.value)} required className="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-white focus:outline-none focus:border-pink-500" />
            </div>
          )}
          <div>
            <label className="block text-sm mb-1">Email</label>
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required className="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-white focus:outline-none focus:border-pink-500" />
          </div>
          <div>
            <label className="block text-sm mb-1">Password</label>
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required className="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-white focus:outline-none focus:border-pink-500" />
          </div>
          <button type="submit" className="w-full bg-pink-600 hover:bg-pink-500 text-white font-medium py-3 rounded-lg transition">{isLogin ? 'Sign In' : 'Continue to Profile Setup'}</button>
        </form>
        <p className="text-center text-sm text-slate-400 mt-6 cursor-pointer" onClick={() => setIsLogin(!isLogin)}>
          {isLogin ? "Don't have an account? Sign Up" : "Already have an account? Sign In"}
        </p>
      </div>
    </div>
  );
};
