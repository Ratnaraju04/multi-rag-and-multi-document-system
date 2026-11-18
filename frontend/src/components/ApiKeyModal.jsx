import React, { useState } from 'react';
import { X, KeyRound, CheckCircle2, AlertCircle } from 'lucide-react';

export default function ApiKeyModal({ isOpen, onClose, onKeySaved }) {
  const [provider, setProvider] = useState('gemini');
  const [apiKey, setApiKey] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);

  if (!isOpen) return null;

  const handleSave = async () => {
    if (!apiKey.trim()) return;
    setLoading(true);
    setMessage(null);
    try {
      const res = await fetch('/api/config/api-key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider, api_key: apiKey.trim() })
      });
      const data = await res.json();
      setMessage({ type: 'success', text: data.message });
      setApiKey('');
      if (onKeySaved) onKeySaved();
      setTimeout(() => {
        onClose();
        setMessage(null);
      }, 1500);
    } catch (err) {
      setMessage({ type: 'error', text: err.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in">
      <div 
        className="relative max-w-md w-full bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl p-6 space-y-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center space-x-2">
            <KeyRound className="w-5 h-5 text-amber-400" />
            <h3 className="text-sm font-bold text-white">API Key Configuration</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {message && (
          <div className={`p-3 rounded-xl border text-xs flex items-center space-x-2 ${
            message.type === 'success'
              ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
              : 'bg-rose-950/40 border-rose-500/40 text-rose-300'
          }`}>
            {message.type === 'success' ? <CheckCircle2 className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
            <span>{message.text}</span>
          </div>
        )}

        <div className="space-y-3 text-xs">
          <div>
            <label className="block text-slate-300 font-semibold mb-1">Select Provider:</label>
            <select
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-slate-100"
            >
              <option value="gemini">Google Gemini (Recommended / Free)</option>
              <option value="groq">Groq Cloud</option>
              <option value="huggingface">HuggingFace Token</option>
            </select>
          </div>

          <div>
            <label className="block text-slate-300 font-semibold mb-1">API Key / Token:</label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Paste API Key here..."
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-slate-100"
            />
            {provider === 'gemini' && (
              <p className="text-[11px] text-slate-500 mt-1">
                Get a free key instantly at <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noreferrer" className="text-blue-400 underline">Google AI Studio</a>.
              </p>
            )}
          </div>
        </div>

        <div className="flex justify-end space-x-2 pt-2 border-t border-slate-800">
          <button
            onClick={onClose}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-slate-300 hover:bg-slate-700"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={loading || !apiKey.trim()}
            className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-50"
          >
            {loading ? 'Saving...' : 'Save API Key'}
          </button>
        </div>
      </div>
    </div>
  );
}
