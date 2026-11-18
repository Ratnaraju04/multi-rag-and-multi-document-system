import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  Play, 
  CheckCircle2, 
  AlertCircle, 
  Clock, 
  ShieldCheck, 
  Sparkles, 
  RefreshCw,
  FileCheck
} from 'lucide-react';

export default function EvaluationDashboard({ status }) {
  const [mode, setMode] = useState('Ground-Truth');
  const [questionsInput, setQuestionsInput] = useState(
    "What is the primary revenue source in the financial report?\n" +
    "What chunk size is recommended for technical documents in AI guidelines?"
  );
  const [answersInput, setAnswersInput] = useState(
    "Quantum Server X1 was the highest revenue product with $1.48M in Q2 and 18.4% growth.\n" +
    "Recommended chunk sizes are 500-1000 characters with 10-15% overlap."
  );

  const [loading, setLoading] = useState(false);
  const [evalResult, setEvalResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);

  const fetchHistory = async () => {
    try {
      const res = await fetch('/api/eval/history');
      const data = await res.json();
      setHistory(data.history || []);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleRunEvaluation = async () => {
    const qList = questionsInput.split('\n').map(q => q.trim()).filter(Boolean);
    const aList = answersInput.split('\n').map(a => a.trim()).filter(Boolean);

    if (qList.length === 0) return;

    setLoading(true);
    setError(null);
    setEvalResult(null);

    try {
      const res = await fetch('/api/eval/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode,
          questions: qList,
          expected_answers: aList
        })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Evaluation failed');
      setEvalResult(data);
      fetchHistory();
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
            <BarChart3 className="w-7 h-7 text-emerald-400" />
            <span>RAG Evaluation & Quality Suite</span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Assess Answer Correctness, Context Precision, and Faithfulness against ground-truth benchmarks.
          </p>
        </div>

        <button
          onClick={handleRunEvaluation}
          disabled={loading || status?.total_chunks === 0}
          className="flex items-center space-x-2 px-5 py-2.5 rounded-xl font-bold text-xs bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white shadow-lg shadow-emerald-600/30 transition-all disabled:opacity-50"
        >
          {loading ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin text-white" />
              <span>Evaluating RAG Quality...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 text-white fill-white" />
              <span>Run Quality Evaluation</span>
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

      {/* ── Mode Selection ── */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <h2 className="text-sm font-bold text-white">Evaluation Mode</h2>
        <div className="flex space-x-3">
          <button
            onClick={() => setMode('Ground-Truth')}
            className={`px-4 py-2 rounded-xl text-xs font-semibold border transition-all ${
              mode === 'Ground-Truth'
                ? 'bg-blue-600 border-blue-500 text-white shadow-md'
                : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-white'
            }`}
          >
            Ground-Truth (Expected Answers)
          </button>
          <button
            onClick={() => setMode('Reference-Free')}
            className={`px-4 py-2 rounded-xl text-xs font-semibold border transition-all ${
              mode === 'Reference-Free'
                ? 'bg-blue-600 border-blue-500 text-white shadow-md'
                : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-white'
            }`}
          >
            Reference-Free (Context Faithfulness Only)
          </button>
        </div>

        {mode === 'Ground-Truth' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Test Questions (One per line):
              </label>
              <textarea
                rows={4}
                value={questionsInput}
                onChange={(e) => setQuestionsInput(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-blue-500 font-mono"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Expected Ground-Truth Answers (One per line, same order):
              </label>
              <textarea
                rows={4}
                value={answersInput}
                onChange={(e) => setAnswersInput(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-blue-500 font-mono"
              />
            </div>
          </div>
        ) : (
          <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-400">
            Reference-free mode assesses answer hallucination rates using only retrieved context chunks without requiring manual golden answers.
          </div>
        )}
      </div>

      {/* ── Evaluation Metrics Scorecards ── */}
      {evalResult && (
        <div className="space-y-4">
          <h2 className="text-base font-bold text-white flex items-center space-x-2">
            <Sparkles className="w-5 h-5 text-amber-400" />
            <span>Latest Evaluation Metrics</span>
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800">
              <span className="text-xs text-slate-400">Answer Correctness</span>
              <div className="text-3xl font-extrabold text-emerald-400 mt-2">
                {((evalResult.aggregate_metrics?.avg_answer_correctness || 0) * 100).toFixed(1)}%
              </div>
              <p className="text-[11px] text-slate-500 mt-1">Semantic overlap with expected answer</p>
            </div>

            <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800">
              <span className="text-xs text-slate-400">Context Precision</span>
              <div className="text-3xl font-extrabold text-blue-400 mt-2">
                {((evalResult.aggregate_metrics?.avg_context_precision || 0) * 100).toFixed(1)}%
              </div>
              <p className="text-[11px] text-slate-500 mt-1">Relevance of retrieved chunks to question</p>
            </div>

            <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800">
              <span className="text-xs text-slate-400">Faithfulness</span>
              <div className="text-3xl font-extrabold text-purple-400 mt-2">
                {((evalResult.aggregate_metrics?.avg_faithfulness || 0) * 100).toFixed(1)}%
              </div>
              <p className="text-[11px] text-slate-500 mt-1">Groundedness in retrieved context</p>
            </div>
          </div>
        </div>
      )}

      {/* ── Historical Runs Table ── */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h2 className="text-base font-bold text-white mb-4">Historical Evaluation Runs ({history.length})</h2>

        {history.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead className="bg-slate-950 text-slate-400 uppercase tracking-wider text-[10px]">
                <tr>
                  <th className="px-4 py-3 rounded-l-lg">Run ID</th>
                  <th className="px-4 py-3">Timestamp</th>
                  <th className="px-4 py-3">Mode</th>
                  <th className="px-4 py-3">Correctness</th>
                  <th className="px-4 py-3">Precision</th>
                  <th className="px-4 py-3 rounded-r-lg">Faithfulness</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 text-slate-300">
                {history.map((h, i) => {
                  const m = h.aggregate_metrics || {};
                  return (
                    <tr key={i} className="hover:bg-slate-800/40">
                      <td className="px-4 py-3 font-mono text-slate-400">{h.run_id || `run-${i}`}</td>
                      <td className="px-4 py-3">{h.timestamp}</td>
                      <td className="px-4 py-3">
                        <span className="px-2 py-0.5 rounded text-[10px] bg-blue-500/10 text-blue-300">
                          {h.mode || 'Ground-Truth'}
                        </span>
                      </td>
                      <td className="px-4 py-3 font-bold text-emerald-400">
                        {((m.avg_answer_correctness || 0) * 100).toFixed(1)}%
                      </td>
                      <td className="px-4 py-3 font-bold text-blue-400">
                        {((m.avg_context_precision || 0) * 100).toFixed(1)}%
                      </td>
                      <td className="px-4 py-3 font-bold text-purple-400">
                        {((m.avg_faithfulness || 0) * 100).toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8 text-xs text-slate-500">
            No evaluation runs recorded yet. Click "Run Quality Evaluation" above to benchmark your RAG pipeline!
          </div>
        )}
      </div>

    </div>
  );
}
