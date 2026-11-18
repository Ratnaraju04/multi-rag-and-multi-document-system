import React, { useState, useEffect } from 'react';
import { 
  Sliders, 
  Layers, 
  Database, 
  Dna, 
  Bot, 
  Cpu, 
  Search, 
  BarChart3, 
  CheckCircle2, 
  AlertCircle, 
  RefreshCw, 
  Zap, 
  Sparkles,
  ChevronDown,
  ChevronRight,
  ShieldCheck,
  Check
} from 'lucide-react';

export default function ConfigModules({ status, onRefreshStatus }) {
  const [activeModule, setActiveModule] = useState(1);
  const [feedback, setFeedback] = useState(null);
  const [loading, setLoading] = useState(false);

  // Module 1: Chunking State
  const [chunkStrategy, setChunkStrategy] = useState(status?.chunking_strategy || 'RecursiveCharacterTextSplitter');
  const [chunkSize, setChunkSize] = useState(status?.chunk_size || 1000);
  const [chunkOverlap, setChunkOverlap] = useState(status?.chunk_overlap || 150);
  const [minChunkSize, setMinChunkSize] = useState(100);
  const [preserveSeparators, setPreserveSeparators] = useState(true);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.5);
  const [previewText, setPreviewText] = useState(
    "Multi-Modal Retrieval-Augmented Generation enhances LLM factuality. " +
    "It splits documents into discrete chunks, encodes them into high-dimensional vectors, " +
    "and retrieves nearest neighbours during query time. Modalities include text, tables, images, and videos."
  );
  const [previewChunks, setPreviewChunks] = useState([]);

  // Module 2: Vector DB State
  const [selectedDb, setSelectedDb] = useState(status?.active_db || 'ChromaDB');
  const [cloudUrl, setCloudUrl] = useState('');
  const [cloudKey, setCloudKey] = useState('');
  const [dbTestResult, setDbTestResult] = useState(null);

  // Module 3: Embedding Model State
  const [selectedEmbed, setSelectedEmbed] = useState(status?.embedding_model || 'sentence-transformers/all-MiniLM-L6-v2');
  const [embedTestResult, setEmbedTestResult] = useState(null);

  // Module 4: Multi-LLM Provider State
  const [provider, setProvider] = useState(status?.llm_provider || 'gemini');
  const [modelName, setModelName] = useState(status?.llm_model || 'gemini-1.5-flash');
  const [apiKeyInput, setApiKeyInput] = useState('');
  const [ollamaUrl, setOllamaUrl] = useState('http://localhost:11434');
  const [llmTestResult, setLlmTestResult] = useState(null);

  // Module 5: Generation Parameters State
  const [temperature, setTemperature] = useState(0.2);
  const [topP, setTopP] = useState(0.9);
  const [topKToken, setTopKToken] = useState(40);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [freqPenalty, setFreqPenalty] = useState(0.0);
  const [presPenalty, setPresPenalty] = useState(0.0);

  // Module 6: Retrieval & Reranker State
  const [retrievalTopK, setRetrievalTopK] = useState(status?.retrieval_top_k || 5);
  const [retThreshold, setRetThreshold] = useState(0.5);
  const [searchStrategy, setSearchStrategy] = useState('similarity');
  const [mmrLambda, setMmrLambda] = useState(0.5);
  const [contextBudget, setContextBudget] = useState(4096);
  const [rerankerEnabled, setRerankerEnabled] = useState(status?.reranker_enabled || false);
  const [rerankerModel, setRerankerModel] = useState('cross-encoder/ms-marco-MiniLM-L-6-v2');
  const [rerankerTopN, setRerankerTopN] = useState(3);

  const chunkStrategies = [
    'RecursiveCharacterTextSplitter',
    'CharacterTextSplitter',
    'TokenTextSplitter',
    'MarkdownHeaderTextSplitter',
    'SemanticChunker'
  ];

  const dbOptions = ['ChromaDB', 'FAISS', 'Qdrant', 'Milvus', 'Pinecone', 'Weaviate'];

  const embeddingOptions = [
    { name: 'sentence-transformers/all-MiniLM-L6-v2', provider: 'HuggingFace', dims: 384, mode: 'Local CPU/GPU' },
    { name: 'BAAI/bge-small-en-v1.5', provider: 'HuggingFace', dims: 384, mode: 'Local CPU/GPU' },
    { name: 'BAAI/bge-base-en-v1.5', provider: 'HuggingFace', dims: 768, mode: 'Local CPU/GPU' },
    { name: 'intfloat/e5-base-v2', provider: 'HuggingFace', dims: 768, mode: 'Local CPU/GPU' },
    { name: 'embedding-001', provider: 'Google Gemini', dims: 768, mode: 'Cloud API' }
  ];

  const llmProviders = [
    { id: 'gemini', name: 'Google Gemini', models: ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash'], hosting: 'CLOUD', license: 'COMMERCIAL' },
    { id: 'ollama', name: 'Ollama (Local)', models: ['llama3.2', 'mistral', 'qwen2.5:7b', 'phi3'], hosting: 'LOCAL', license: 'OPEN-WEIGHT' },
    { id: 'groq', name: 'Groq Cloud', models: ['llama-3.3-70b-versatile', 'mixtral-8x7b-32768', 'gemma2-9b-it'], hosting: 'CLOUD', license: 'FAST LPU' },
    { id: 'huggingface', name: 'HuggingFace Hub', models: ['meta-llama/Llama-3-8B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.3'], hosting: 'CLOUD', license: 'INFERENCE API' }
  ];

  // Handle updates
  const handleSaveChunking = async () => {
    setLoading(true);
    setFeedback(null);
    try {
      const res = await fetch('/api/config/chunking', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy: chunkStrategy,
          chunk_size: chunkSize,
          chunk_overlap: chunkOverlap,
          min_chunk_size: minChunkSize,
          preserve_separators: preserveSeparators,
          similarity_threshold: similarityThreshold
        })
      });
      const data = await res.json();
      setFeedback({ type: 'success', message: 'Chunking configuration updated.' });
      onRefreshStatus();
    } catch (err) {
      setFeedback({ type: 'error', message: err.message });
    } finally {
      setLoading(false);
    }
  };

  const handlePreviewChunks = async () => {
    if (!previewText.trim()) return;
    setLoading(true);
    try {
      const res = await fetch('/api/preview-chunks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: previewText,
          strategy: chunkStrategy,
          chunk_size: chunkSize,
          chunk_overlap: chunkOverlap
        })
      });
      const data = await res.json();
      setPreviewChunks(data.chunks || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleTestDb = async () => {
    setLoading(true);
    setDbTestResult(null);
    try {
      const res = await fetch('/api/test/db', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          db_name: selectedDb,
          cloud_url: cloudUrl,
          cloud_key: cloudKey
        })
      });
      const data = await res.json();
      setDbTestResult(data);
    } catch (err) {
      setDbTestResult({ success: false, message: err.message });
    } finally {
      setLoading(false);
    }
  };

  const handleTestEmbedding = async () => {
    setLoading(true);
    setEmbedTestResult(null);
    try {
      const res = await fetch('/api/test/embedding', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: selectedEmbed })
      });
      const data = await res.json();
      setEmbedTestResult(data);
      onRefreshStatus();
    } catch (err) {
      setEmbedTestResult({ success: false, message: err.message });
    } finally {
      setLoading(false);
    }
  };

  const handleTestLlm = async () => {
    setLoading(true);
    setLlmTestResult(null);
    try {
      const res = await fetch('/api/test/llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider,
          model_name: modelName,
          api_key: apiKeyInput,
          base_url: ollamaUrl
        })
      });
      const data = await res.json();
      setLlmTestResult(data);
    } catch (err) {
      setLlmTestResult({ success: false, message: err.message });
    } finally {
      setLoading(false);
    }
  };

  const handleSaveLlm = async () => {
    setLoading(true);
    try {
      await fetch('/api/config/llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider,
          model_name: modelName,
          api_key: apiKeyInput,
          base_url: ollamaUrl
        })
      });
      setFeedback({ type: 'success', message: `Active LLM set to ${provider.toUpperCase()}: ${modelName}` });
      onRefreshStatus();
    } catch (err) {
      setFeedback({ type: 'error', message: err.message });
    } finally {
      setLoading(false);
    }
  };

  const handleResetFactualDefaults = async () => {
    setTemperature(0.2);
    setTopP(0.9);
    setMaxTokens(2048);
    setFreqPenalty(0.0);
    setPresPenalty(0.0);
    try {
      await fetch('/api/config/generation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          temperature: 0.2,
          top_p: 0.9,
          max_output_tokens: 2048,
          frequency_penalty: 0.0,
          presence_penalty: 0.0
        })
      });
      setFeedback({ type: 'success', message: 'Reset generation parameters to factual defaults.' });
    } catch (err) {
      console.error(err);
    }
  };

  const handleSaveRetrieval = async () => {
    setLoading(true);
    try {
      await fetch('/api/config/retrieval', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          top_k: retrievalTopK,
          similarity_threshold: retThreshold,
          search_type: searchStrategy,
          mmr_lambda: mmrLambda,
          max_context_tokens: contextBudget,
          reranker_enabled: rerankerEnabled,
          reranker_model: rerankerModel,
          reranker_top_n: rerankerTopN
        })
      });
      setFeedback({ type: 'success', message: 'Retrieval & Reranking configuration updated.' });
      onRefreshStatus();
    } catch (err) {
      setFeedback({ type: 'error', message: err.message });
    } finally {
      setLoading(false);
    }
  };

  const currentModels = llmProviders.find(p => p.id === provider)?.models || [];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      
      {/* ── Page Header ── */}
      <div>
        <h1 className="text-2xl font-extrabold text-white tracking-tight flex items-center space-x-2">
          <Sliders className="w-7 h-7 text-indigo-400" />
          <span>Advanced Configuration Studio</span>
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Configure and fine-tune all 7 modules of your enterprise Multi-Modal RAG pipeline.
        </p>
      </div>

      {feedback && (
        <div className={`p-4 rounded-xl border flex items-center space-x-3 text-sm ${
          feedback.type === 'success' 
            ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300' 
            : 'bg-rose-950/40 border-rose-500/40 text-rose-300'
        }`}>
          {feedback.type === 'success' ? (
            <CheckCircle2 className="w-5 h-5 shrink-0 text-emerald-400" />
          ) : (
            <AlertCircle className="w-5 h-5 shrink-0 text-rose-400" />
          )}
          <span>{feedback.message}</span>
        </div>
      )}

      {/* ── Modules Navigation Bar ── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2">
        {[
          { num: 1, label: 'Chunking', icon: Layers },
          { num: 2, label: 'Vector DB', icon: Database },
          { num: 3, label: 'Embeddings', icon: Dna },
          { num: 4, label: 'LLM Engine', icon: Bot },
          { num: 5, label: 'Generation', icon: Cpu },
          { num: 6, label: 'Retrieval', icon: Search },
          { num: 7, label: 'Eval Status', icon: BarChart3 },
        ].map((m) => {
          const Icon = m.icon;
          const isActive = activeModule === m.num;
          return (
            <button
              key={m.num}
              onClick={() => {
                setActiveModule(m.num);
                setFeedback(null);
              }}
              className={`flex flex-col items-center justify-center p-3 rounded-xl border text-xs font-semibold transition-all ${
                isActive
                  ? 'bg-blue-600 border-blue-500 text-white shadow-lg shadow-blue-600/30 ring-1 ring-white/20'
                  : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-white hover:bg-slate-800/80'
              }`}
            >
              <Icon className="w-4 h-4 mb-1" />
              <span>Module {m.num}</span>
              <span className="text-[10px] font-normal opacity-80">{m.label}</span>
            </button>
          );
        })}
      </div>

      {/* ════════════════════════════════════════════════════════════════════ */}
      {/* MODULE 1: Advanced Chunking Settings                                */}
      {/* ════════════════════════════════════════════════════════════════════ */}
      {activeModule === 1 && (
        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <h2 className="text-lg font-bold text-white flex items-center space-x-2">
                <Layers className="w-5 h-5 text-blue-400" />
                <span>Module 1: Chunking & Text Splitting Settings</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Configure document chunking strategies, overlap window, and boundary preservation.
              </p>
            </div>
            <button
              onClick={handleSaveChunking}
              disabled={loading}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold rounded-xl transition-all shadow-md"
            >
              Save Chunking Settings
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Chunking Algorithm Strategy:
                </label>
                <select
                  value={chunkStrategy}
                  onChange={(e) => setChunkStrategy(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-blue-500"
                >
                  {chunkStrategies.map((st, i) => (
                    <option key={i} value={st}>{st}</option>
                  ))}
                </select>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300 font-medium">Chunk Size (Characters):</span>
                  <span className="font-bold text-blue-400">{chunkSize} chars</span>
                </div>
                <input
                  type="range"
                  min="100"
                  max="4000"
                  step="50"
                  value={chunkSize}
                  onChange={(e) => setChunkSize(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300 font-medium">Chunk Overlap (Characters):</span>
                  <span className="font-bold text-purple-400">{chunkOverlap} chars</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="500"
                  step="10"
                  value={chunkOverlap}
                  onChange={(e) => setChunkOverlap(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
                />
              </div>

              <div className="flex items-center space-x-2 pt-2">
                <input
                  type="checkbox"
                  id="preserveSep"
                  checked={preserveSeparators}
                  onChange={(e) => setPreserveSeparators(e.target.checked)}
                  className="rounded border-slate-700 text-blue-600 focus:ring-0"
                />
                <label htmlFor="preserveSep" className="text-xs text-slate-300 cursor-pointer">
                  Preserve Natural Sentence / Paragraph Separators
                </label>
              </div>
            </div>

            {/* Live Interactive Chunk Previewer */}
            <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 flex flex-col">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-slate-200">Interactive Preview Playground:</span>
                <button
                  onClick={handlePreviewChunks}
                  className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs rounded-lg font-medium transition-colors"
                >
                  Split & Preview Chunks
                </button>
              </div>
              <textarea
                rows={3}
                value={previewText}
                onChange={(e) => setPreviewText(e.target.value)}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 focus:outline-none focus:border-blue-500 mb-3"
                placeholder="Type or paste text to test chunk boundaries..."
              />

              {previewChunks.length > 0 && (
                <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                  <span className="text-[11px] font-semibold text-emerald-400">
                    Generated {previewChunks.length} chunks with strategy `{chunkStrategy}`:
                  </span>
                  {previewChunks.map((c) => (
                    <div key={c.index} className="p-2 bg-slate-900 border border-slate-800 rounded-lg text-xs font-mono">
                      <div className="text-[10px] text-slate-400 mb-1">
                        Chunk #{c.index} ({c.chars} chars)
                      </div>
                      <p className="text-slate-300">{c.text}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════════════ */}
      {/* MODULE 2: Vector Database Management                                */}
      {/* ════════════════════════════════════════════════════════════════════ */}
      {activeModule === 2 && (
        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <h2 className="text-lg font-bold text-white flex items-center space-x-2">
                <Database className="w-5 h-5 text-purple-400" />
                <span>Module 2: Vector Database Management</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Switch active vector stores, test indexing connectivity, and monitor chunk counts.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Select Active Vector Store:
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {dbOptions.map((db, i) => (
                    <button
                      key={i}
                      onClick={() => setSelectedDb(db)}
                      className={`p-3 rounded-xl border text-left text-xs font-semibold transition-all ${
                        selectedDb === db
                          ? 'bg-purple-600/20 border-purple-500 text-purple-200 ring-1 ring-purple-500'
                          : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span>{db}</span>
                        {selectedDb === db && <Check className="w-3.5 h-3.5 text-purple-400" />}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {['Qdrant', 'Milvus', 'Pinecone', 'Weaviate'].includes(selectedDb) && (
                <div className="space-y-3 p-3 bg-slate-950 rounded-xl border border-slate-800">
                  <span className="text-xs font-semibold text-slate-300">Cloud Provider Endpoint & Credentials:</span>
                  <input
                    type="text"
                    value={cloudUrl}
                    onChange={(e) => setCloudUrl(e.target.value)}
                    placeholder="Endpoint URL (e.g. https://xyz.cloud.provider:6333)"
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200"
                  />
                  <input
                    type="password"
                    value={cloudKey}
                    onChange={(e) => setCloudKey(e.target.value)}
                    placeholder="API Key"
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200"
                  />
                </div>
              )}

              <button
                onClick={handleTestDb}
                disabled={loading}
                className="flex items-center space-x-1.5 px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white text-xs font-semibold rounded-xl shadow-md transition-all"
              >
                <Zap className="w-3.5 h-3.5" />
                <span>Test DB Connectivity</span>
              </button>

              {dbTestResult && (
                <div className={`p-3 rounded-xl border text-xs flex items-center space-x-2 ${
                  dbTestResult.success
                    ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
                    : 'bg-rose-950/40 border-rose-500/40 text-rose-300'
                }`}>
                  {dbTestResult.success ? <CheckCircle2 className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
                  <span>{dbTestResult.message}</span>
                </div>
              )}
            </div>

            {/* Active Database Stats Card */}
            <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-5 flex flex-col justify-between">
              <div>
                <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Active Store Overview</span>
                <h3 className="text-xl font-extrabold text-white mt-1">{status?.active_db}</h3>
                <div className="mt-4 p-4 rounded-xl bg-slate-900 border border-slate-800">
                  <div className="text-xs text-slate-400">Total Indexed Vector Chunks:</div>
                  <div className="text-3xl font-extrabold text-blue-400 mt-1">
                    {status?.total_chunks || 0}
                  </div>
                </div>
              </div>
              <p className="text-[11px] text-slate-500 mt-4">
                ChromaDB and FAISS run seamlessly on your local environment without external cloud dependencies.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════════════ */}
      {/* MODULE 3: Embedding Model Hub                                       */}
      {/* ════════════════════════════════════════════════════════════════════ */}
      {activeModule === 3 && (
        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <h2 className="text-lg font-bold text-white flex items-center space-x-2">
                <Dna className="w-5 h-5 text-emerald-400" />
                <span>Module 3: Embedding Model Hub</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Select semantic vector models for dense encoding across text, table, and multi-modal descriptions.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <label className="block text-xs font-semibold text-slate-300">
                Choose Dense Embedding Model:
              </label>
              <div className="space-y-2">
                {embeddingOptions.map((emb, idx) => (
                  <div
                    key={idx}
                    onClick={() => setSelectedEmbed(emb.name)}
                    className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                      selectedEmbed === emb.name
                        ? 'bg-emerald-600/15 border-emerald-500 text-emerald-200'
                        : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold text-xs text-slate-200">{emb.name}</span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-300">
                        {emb.dims}d
                      </span>
                    </div>
                    <div className="flex items-center space-x-2 text-[11px] text-slate-400">
                      <span>Provider: {emb.provider}</span>
                      <span>•</span>
                      <span>Mode: {emb.mode}</span>
                    </div>
                  </div>
                ))}
              </div>

              <button
                onClick={handleTestEmbedding}
                disabled={loading}
                className="flex items-center space-x-1.5 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded-xl shadow-md transition-all"
              >
                <Zap className="w-3.5 h-3.5" />
                <span>Test Embedding Function Latency</span>
              </button>

              {embedTestResult && (
                <div className={`p-3 rounded-xl border text-xs flex items-center space-x-2 ${
                  embedTestResult.success
                    ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
                    : 'bg-rose-950/40 border-rose-500/40 text-rose-300'
                }`}>
                  {embedTestResult.success ? <CheckCircle2 className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
                  <span>{embedTestResult.message}</span>
                </div>
              )}
            </div>

            {/* Embedding Details Card */}
            <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-5 flex flex-col justify-between">
              <div>
                <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Active Model Spec</span>
                <h3 className="text-lg font-bold text-white mt-1 break-all">{selectedEmbed}</h3>
                
                <div className="grid grid-cols-2 gap-3 mt-4">
                  <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
                    <span className="text-[11px] text-slate-400">Vector Dimensions:</span>
                    <div className="text-lg font-bold text-emerald-400">
                      {embeddingOptions.find(e => e.name === selectedEmbed)?.dims || 384}d
                    </div>
                  </div>
                  <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
                    <span className="text-[11px] text-slate-400">Execution Mode:</span>
                    <div className="text-sm font-bold text-blue-400 mt-1">
                      {embeddingOptions.find(e => e.name === selectedEmbed)?.mode || 'Local'}
                    </div>
                  </div>
                </div>
              </div>
              <p className="text-[11px] text-slate-500 mt-4">
                Vectors are normalized for cosine similarity distance during index retrieval.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════════════ */}
      {/* MODULE 4: Multi-LLM Providers & Models                              */}
      {/* ════════════════════════════════════════════════════════════════════ */}
      {activeModule === 4 && (
        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <h2 className="text-lg font-bold text-white flex items-center space-x-2">
                <Bot className="w-5 h-5 text-blue-400" />
                <span>Module 4: Multi-LLM Provider & Model Engine</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Switch between Gemini, Ollama local models, Groq LPUs, and Hugging Face inference.
              </p>
            </div>
            <button
              onClick={handleSaveLlm}
              disabled={loading}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold rounded-xl transition-all shadow-md"
            >
              Set as Active Pipeline LLM
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              {/* Provider Selection */}
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Select LLM Provider:
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {llmProviders.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => {
                        setProvider(p.id);
                        setModelName(p.models[0]);
                        setLlmTestResult(null);
                      }}
                      className={`p-3 rounded-xl border text-left text-xs font-semibold transition-all ${
                        provider === p.id
                          ? 'bg-blue-600/20 border-blue-500 text-blue-200 ring-1 ring-blue-500'
                          : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span>{p.name}</span>
                        {provider === p.id && <Check className="w-3.5 h-3.5 text-blue-400" />}
                      </div>
                      <div className="flex space-x-1 mt-1">
                        <span className="text-[9px] px-1 py-0.2 rounded bg-slate-800 text-slate-400">
                          {p.hosting}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Model Selection */}
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Select Model:
                </label>
                <select
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-blue-500"
                >
                  {currentModels.map((m, i) => (
                    <option key={i} value={m}>{m}</option>
                  ))}
                </select>
              </div>

              {/* Provider Config Fields */}
              {provider === 'gemini' && (
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">
                    Gemini API Key (leave empty to use system environment key):
                  </label>
                  <input
                    type="password"
                    value={apiKeyInput}
                    onChange={(e) => setApiKeyInput(e.target.value)}
                    placeholder="AIzaSy..."
                    className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-100"
                  />
                </div>
              )}

              {provider === 'ollama' && (
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">
                    Ollama Base URL:
                  </label>
                  <input
                    type="text"
                    value={ollamaUrl}
                    onChange={(e) => setOllamaUrl(e.target.value)}
                    placeholder="http://localhost:11434"
                    className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-100"
                  />
                </div>
              )}

              {(provider === 'groq' || provider === 'huggingface') && (
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">
                    {provider.toUpperCase()} API Key / Token:
                  </label>
                  <input
                    type="password"
                    value={apiKeyInput}
                    onChange={(e) => setApiKeyInput(e.target.value)}
                    placeholder="gsk_... or hf_..."
                    className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-100"
                  />
                </div>
              )}

              <button
                onClick={handleTestLlm}
                disabled={loading}
                className="flex items-center space-x-1.5 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold rounded-xl border border-slate-700 transition-all"
              >
                <Zap className="w-3.5 h-3.5 text-amber-400" />
                <span>Test LLM API Connection</span>
              </button>

              {llmTestResult && (
                <div className={`p-3 rounded-xl border text-xs flex items-center space-x-2 ${
                  llmTestResult.success
                    ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
                    : 'bg-rose-950/40 border-rose-500/40 text-rose-300'
                }`}>
                  {llmTestResult.success ? <CheckCircle2 className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
                  <span>{llmTestResult.message}</span>
                </div>
              )}
            </div>

            {/* Model Profile Card */}
            <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-5 flex flex-col justify-between">
              <div>
                <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Engine Specification</span>
                <h3 className="text-xl font-extrabold text-white mt-1">{modelName}</h3>
                
                <div className="flex flex-wrap gap-2 mt-3">
                  <span className="px-2.5 py-1 rounded-md text-[11px] font-semibold bg-blue-500/10 text-blue-300 border border-blue-500/20">
                    Provider: {provider.toUpperCase()}
                  </span>
                  <span className="px-2.5 py-1 rounded-md text-[11px] font-semibold bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                    Multimodal Support
                  </span>
                </div>

                <p className="text-xs text-slate-400 mt-4 leading-relaxed">
                  Processes user queries against combined retrieved contexts including extracted text, tabular spreadsheet data, image descriptions, and timestamped video keyframe captions.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════════════ */}
      {/* MODULE 5: Generation Parameters                                     */}
      {/* ════════════════════════════════════════════════════════════════════ */}
      {activeModule === 5 && (
        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <h2 className="text-lg font-bold text-white flex items-center space-x-2">
                <Cpu className="w-5 h-5 text-amber-400" />
                <span>Module 5: Generation Sampling Parameters</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Tune temperature, nucleus sampling, and output token limits for factual RAG responses.
              </p>
            </div>
            <button
              onClick={handleResetFactualDefaults}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-amber-300 border border-amber-500/30 text-xs font-semibold rounded-xl transition-all"
            >
              Reset to Factual Defaults (0.2 Temp)
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300 font-medium">Temperature (Factual vs Creative):</span>
                  <span className="font-bold text-amber-400">{temperature}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="2.0"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300 font-medium">Top-P (Nucleus Sampling):</span>
                  <span className="font-bold text-blue-400">{topP}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={topP}
                  onChange={(e) => setTopP(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300 font-medium">Max Output Tokens:</span>
                  <span className="font-bold text-emerald-400">{maxTokens} tokens</span>
                </div>
                <input
                  type="range"
                  min="256"
                  max="8192"
                  step="256"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
              </div>
            </div>

            <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-5">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">RAG Best Practice</span>
              <p className="text-xs text-slate-300 mt-2 leading-relaxed">
                For Retrieval-Augmented Generation, lower temperatures (<strong className="text-amber-400">0.0 to 0.3</strong>) drastically reduce hallucinations by constraining the language model to synthesize only the retrieved context chunks without speculative completion.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════════════ */}
      {/* MODULE 6: Retrieval & Reranking                                     */}
      {/* ════════════════════════════════════════════════════════════════════ */}
      {activeModule === 6 && (
        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <h2 className="text-lg font-bold text-white flex items-center space-x-2">
                <Search className="w-5 h-5 text-cyan-400" />
                <span>Module 6: Retrieval & Cross-Encoder Reranking</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Set retrieval Top-K, MMR diversity lambda, and cross-encoder precision rerankers.
              </p>
            </div>
            <button
              onClick={handleSaveRetrieval}
              disabled={loading}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold rounded-xl transition-all shadow-md"
            >
              Save Retrieval Settings
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300 font-medium">Retrieval Top-K (Documents fetched):</span>
                  <span className="font-bold text-cyan-400">{retrievalTopK} chunks</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="20"
                  value={retrievalTopK}
                  onChange={(e) => setRetrievalTopK(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300 font-medium">Similarity Distance Threshold:</span>
                  <span className="font-bold text-purple-400">{retThreshold}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={retThreshold}
                  onChange={(e) => setRetThreshold(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
                />
              </div>

              {/* Reranker Toggle */}
              <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-200">Cross-Encoder Precision Reranker:</span>
                  <input
                    type="checkbox"
                    checked={rerankerEnabled}
                    onChange={(e) => setRerankerEnabled(e.target.checked)}
                    className="rounded border-slate-700 text-blue-600 focus:ring-0"
                  />
                </div>

                {rerankerEnabled && (
                  <>
                    <select
                      value={rerankerModel}
                      onChange={(e) => setRerankerModel(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200"
                    >
                      <option value="cross-encoder/ms-marco-MiniLM-L-6-v2">cross-encoder/ms-marco-MiniLM-L-6-v2</option>
                      <option value="cross-encoder/ms-marco-TinyBERT-L-2-v2">cross-encoder/ms-marco-TinyBERT-L-2-v2</option>
                    </select>

                    <div className="flex justify-between text-xs">
                      <span className="text-slate-400">Reranked Top-N passed to LLM:</span>
                      <span className="font-bold text-blue-400">{rerankerTopN} chunks</span>
                    </div>
                  </>
                )}
              </div>
            </div>

            <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-5">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">How Reranking Works</span>
              <p className="text-xs text-slate-300 mt-2 leading-relaxed">
                Standard vector search retrieves candidate chunks fast using bi-encoders. Cross-encoders then perform a joint query-document relevance scoring on the candidates, yielding higher context precision before prompting the LLM.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════════════ */}
      {/* MODULE 7: Quick Evaluation Summary                                  */}
      {/* ════════════════════════════════════════════════════════════════════ */}
      {activeModule === 7 && (
        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-emerald-400" />
            <span>Module 7: Pipeline Status & Health Overview</span>
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
            <div className="p-4 bg-slate-950 rounded-xl border border-slate-800">
              <span className="text-xs text-slate-400">Active Vector DB</span>
              <div className="text-xl font-bold text-white mt-1">{status?.active_db}</div>
              <span className="text-[11px] text-emerald-400">{status?.total_chunks || 0} chunks indexed</span>
            </div>
            <div className="p-4 bg-slate-950 rounded-xl border border-slate-800">
              <span className="text-xs text-slate-400">Dense Embedding Model</span>
              <div className="text-sm font-bold text-white mt-1 truncate">{status?.embedding_model}</div>
              <span className="text-[11px] text-blue-400">Normalized L2 / Cosine</span>
            </div>
            <div className="p-4 bg-slate-950 rounded-xl border border-slate-800">
              <span className="text-xs text-slate-400">Active LLM Provider</span>
              <div className="text-xl font-bold text-white mt-1">{status?.llm_provider?.toUpperCase()}</div>
              <span className="text-[11px] text-purple-400">{status?.llm_model}</span>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
