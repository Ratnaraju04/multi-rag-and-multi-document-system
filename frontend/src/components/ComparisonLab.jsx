import React, { useState } from 'react';
import { 
  FlaskConical, 
  Play, 
  Layers, 
  Clock, 
  FileText, 
  CheckCircle2, 
  AlertCircle, 
  RefreshCw,
  Sparkles
} from 'lucide-react';

export default function ComparisonLab({ status }) {
  const [questionsInput, setQuestionsInput] = useState(
    "What is the primary revenue source in the financial report and what was its growth rate?\n" +
    "What chunk size is recommended for technical documents in the AI guidelines?\n" +
    "Describe the visual components shown in the architecture diagram."
  );

  const [configs, setConfigs] = useState([
    { label: 'Config A (Gemini 1.5 Flash)', provider: 'gemini', model_name: 'gemini-1.5-flash', top_k: 5, reranker_enabled: false },
    { label: 'Config B (Gemini 2.0 Flash + Rerank)', provider: 'gemini', model_name: 'gemini-2.0-flash', top_k: 8, reranker_enabled: true },
  ]);

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const addConfig = () => {
    if (configs.length >= 4) return;
    const nextChar = String.fromCharCode(65 + configs.length);
    setConfigs(prev => [
      ...prev,
      {
        label: `Config ${nextChar} (Gemini 1.5 Pro)`,
        provider: 'gemini',
        model_name: 'gemini-1.5-pro',
        top_k: 5,
        reranker_enabled: false
      }
    ]);
  };

  const removeConfig = (idx) => {
    if (configs.length <= 2) return;
    setConfigs(prev => prev.filter((_, i) => i !== idx));
  };

  const updateConfig = (idx, field, value) => {
    setConfigs(prev => {
      const copy = [...prev];
      copy[idx] = { ...copy[idx], [field]: value };
      return copy;
    });
  };

  const handleRunComparison = async () => {
    const qList = questionsInput.split('\n').map(q => q.trim()).filter(Boolean);
    if (qList.length === 0) return;

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const res = await fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          questions: qList,
          configs
        })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Comparison run failed');
      setResults(data.benchmark_results || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      
      {/* ── Header ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold text-white tracking-tight flex items-center space-x-2">
            <FlaskConical className="w-7 h-7 text-indigo-400" />
            <span>RAG Comparison & Experimentation Lab</span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Benchmark different LLMs, retrieval Top-K values, and cross-encoder rerankers side-by-side under identical queries.
          </p>
        </div>

        <button
          onClick={handleRunComparison}
          disabled={loading || status?.total_chunks === 0}
          className="flex items-center space-x-2 px-5 py-2.5 rounded-xl font-bold text-xs bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-lg shadow-blue-600/30 transition-all disabled:opacity-50"
        >
          {loading ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin text-white" />
              <span>Running Benchmark...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 text-white fill-white" />
              <span>Run Comparison Experiment</span>
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-xl border bg-rose-950/40 border-rose-500/40 text-rose-300 flex items-center space-x-3 text-sm">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* ── Step 1: Benchmark Questions ── */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h2 className="text-sm font-bold text-white mb-2 flex items-center space-x-2">
          <span className="w-5 h-5 rounded-full bg-blue-600 text-white text-[11px] flex items-center justify-center font-bold">1</span>
          <span>Shared Benchmark Questions (One per line)</span>
        </h2>
        <textarea
          rows={3}
          value={questionsInput}
          onChange={(e) => setQuestionsInput(e.target.value)}
          placeholder="Enter benchmark queries..."
          className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-blue-500 font-mono"
        />
      </div>

      {/* ── Step 2: Configure Pipelines ── */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-bold text-white flex items-center space-x-2">
            <span className="w-5 h-5 rounded-full bg-purple-600 text-white text-[11px] flex items-center justify-center font-bold">2</span>
            <span>Pipeline Configurations ({configs.length}/4)</span>
          </h2>
          {configs.length < 4 && (
            <button
              onClick={addConfig}
              className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded-lg text-xs font-semibold transition-colors"
            >
              + Add Pipeline
            </button>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {configs.map((cfg, idx) => (
            <div key={idx} className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-white">{cfg.label}</span>
                {configs.length > 2 && (
                  <button
                    onClick={() => removeConfig(idx)}
                    className="text-[11px] text-rose-400 hover:underline"
                  >
                    Remove
                  </button>
                )}
              </div>

              <div>
                <label className="block text-[11px] text-slate-400 mb-1">Provider:</label>
                <select
                  value={cfg.provider}
                  onChange={(e) => updateConfig(idx, 'provider', e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-slate-200"
                >
                  <option value="gemini">Google Gemini</option>
                  <option value="groq">Groq</option>
                  <option value="ollama">Ollama (Local)</option>
                </select>
              </div>

              <div>
                <label className="block text-[11px] text-slate-400 mb-1">Model Name:</label>
                <input
                  type="text"
                  value={cfg.model_name}
                  onChange={(e) => updateConfig(idx, 'model_name', e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-slate-200"
                />
              </div>

              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">Top-K:</span>
                <input
                  type="number"
                  min="1"
                  max="15"
                  value={cfg.top_k}
                  onChange={(e) => updateConfig(idx, 'top_k', parseInt(e.target.value))}
                  className="w-16 bg-slate-900 border border-slate-700 rounded px-2 py-0.5 text-xs text-white"
                />
              </div>

              <div className="flex items-center justify-between text-xs pt-1">
                <span className="text-slate-400">Reranker:</span>
                <input
                  type="checkbox"
                  checked={cfg.reranker_enabled}
                  onChange={(e) => updateConfig(idx, 'reranker_enabled', e.target.checked)}
                  className="rounded border-slate-700 text-blue-600 focus:ring-0"
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Step 3: Comparison Results Grid ── */}
      {results && results.length > 0 && (
        <div className="space-y-6">
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <Sparkles className="w-5 h-5 text-amber-400" />
            <span>Benchmark Comparison Results</span>
          </h2>

          {results.map((item, qIdx) => (
            <div key={qIdx} className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
              <div className="p-3 bg-slate-950 rounded-xl border border-slate-800/80">
                <span className="text-[11px] font-bold uppercase text-blue-400 tracking-wider">Test Query #{qIdx + 1}</span>
                <p className="text-sm font-semibold text-white mt-0.5">{item.question}</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {item.runs.map((run, rIdx) => (
                  <div key={rIdx} className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/80 flex flex-col justify-between">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-bold text-slate-200">{run.config_label}</span>
                        <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-blue-500/20 text-blue-300">
                          {run.latency_seconds}s
                        </span>
                      </div>
                      <p className="text-xs text-slate-300 leading-relaxed whitespace-pre-wrap">
                        {run.answer}
                      </p>
                    </div>

                    <div className="mt-4 pt-2 border-t border-slate-800 flex justify-between text-[11px] text-slate-400">
                      <span>Model: {run.model}</span>
                      <span>Sources: {run.num_sources}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

    </div>
  );
}
