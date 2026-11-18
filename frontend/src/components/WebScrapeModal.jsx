import React, { useState } from 'react';
import { X, Globe, Sparkles, RefreshCw, CheckCircle2, AlertCircle, ArrowRight } from 'lucide-react';

export default function WebScrapeModal({ isOpen, onClose, onScrapeComplete }) {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  if (!isOpen) return null;

  const sampleUrls = [
    { label: 'Wikipedia: Retrieval-Augmented Gen', url: 'https://en.wikipedia.org/wiki/Retrieval-augmented_generation' },
    { label: 'Wikipedia: Large Language Model', url: 'https://en.wikipedia.org/wiki/Large_language_model' },
    { label: 'Wikipedia: Vector Database', url: 'https://en.wikipedia.org/wiki/Vector_database' },
  ];

  const handleScrape = async (targetUrl = url) => {
    const finalUrl = targetUrl.trim();
    if (!finalUrl) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetch('/api/scrape-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: finalUrl })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Scraping failed');

      if (onScrapeComplete) {
        onScrapeComplete(data);
      }
      setUrl('');
      onClose();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div 
        className="relative max-w-lg w-full bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl p-6 space-y-4"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
              <Globe className="w-4 h-4 text-blue-400" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white">Scrape & Index Web Page</h3>
              <p className="text-xs text-slate-400">Extracts clean headings, text, and tables into vectors</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {error && (
          <div className="p-3 rounded-xl border bg-rose-950/40 border-rose-500/40 text-rose-300 text-xs flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
            <span>{error}</span>
          </div>
        )}

        {/* Input */}
        <div className="space-y-3">
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5">
              Target Web URL:
            </label>
            <div className="relative">
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://en.wikipedia.org/wiki/..."
                disabled={loading}
                className="w-full bg-slate-950 border border-slate-700 rounded-xl px-4 py-2.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500 font-mono disabled:opacity-50"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleScrape();
                }}
              />
            </div>
          </div>

          {/* Quick Samples */}
          <div>
            <span className="text-[11px] font-semibold text-slate-400 block mb-1.5">
              Quick Test Articles:
            </span>
            <div className="space-y-1.5">
              {sampleUrls.map((s, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setUrl(s.url);
                    handleScrape(s.url);
                  }}
                  disabled={loading}
                  className="w-full text-left px-3 py-1.5 rounded-lg bg-slate-950/70 hover:bg-slate-800 border border-slate-800/80 text-xs text-slate-300 hover:text-blue-300 flex items-center justify-between transition-colors group"
                >
                  <span className="truncate">{s.label}</span>
                  <ArrowRight className="w-3 h-3 text-slate-500 group-hover:text-blue-400 shrink-0 ml-2" />
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="flex justify-end space-x-2 pt-3 border-t border-slate-800">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-slate-800 text-slate-300 hover:bg-slate-700 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => handleScrape()}
            disabled={loading || !url.trim()}
            className="px-5 py-2 rounded-xl text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white transition-all shadow-md shadow-blue-600/30 flex items-center space-x-1.5 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Scraping & Indexing...</span>
              </>
            ) : (
              <>
                <Globe className="w-3.5 h-3.5" />
                <span>Scrape & Vectorize URL</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
