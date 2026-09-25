import React, { useState } from 'react';
import { RegisterPayload } from '../types/auth';

interface Props {
  initialData: { email: string; pass: string; name: string };
  onComplete: (token: string) => void;
}

export const MotherOnboarding: React.FC<Props> = ({ initialData, onComplete }) => {
  const [formData, setFormData] = useState({
    pregnancy_status: 'pregnant',
    current_week: 1,
    expected_due_date: '2026-08-19',
    pregnancies_count: 1,
    previous_c_sections: 0,
    blood_type: 'O+',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const payload: RegisterPayload = {
      email: initialData.email,
      password: initialData.pass,
      full_name: initialData.name,
      persona: 'mother',
      language: 'ar',
      mother_profile: {
        ...formData,
        chronic_conditions: [],
        allergies: [],
        key_interests: []
      }
    };

    try {
      const res = await fetch('http://localhost:8000/api/v1/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (res.ok) {
        localStorage.setItem('token', data.access_token);
        onComplete(data.access_token);
      } else {
        alert(data.detail || 'Registration failed');
      }
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-slate-900 text-slate-100 p-4">
      <div className="w-full max-w-lg bg-slate-800 border border-slate-700 rounded-2xl p-8 shadow-xl">
        <h2 className="text-2xl font-bold text-center mb-2">Mother Profile Setup</h2>
        <p className="text-sm text-slate-400 text-center mb-6">Help us personalize ELARA for your journey</p>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm mb-1">Current Week</label>
              <input type="number" min="1" max="42" value={formData.current_week} onChange={(e) => setFormData({...formData, current_week: Number(e.target.value)})} className="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-white" />
            </div>
            <div>
              <label className="block text-sm mb-1">Blood Type</label>
              <input type="text" value={formData.blood_type} onChange={(e) => setFormData({...formData, blood_type: e.target.value})} className="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-white" />
            </div>
          </div>
          <div>
            <label className="block text-sm mb-1">Expected Due Date</label>
            <input type="date" value={formData.expected_due_date} onChange={(e) => setFormData({...formData, expected_due_date: e.target.value})} className="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-white" />
          </div>
          <button type="submit" className="w-full bg-pink-600 hover:bg-pink-500 text-white font-medium py-3 rounded-lg transition mt-4">Complete Setup & Start Chat</button>
        </form>
      </div>
    </div>
  );
};
