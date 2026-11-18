import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, 
  Sparkles, 
  Trash2, 
  Bot, 
  User, 
  Image as ImageIcon, 
  Film, 
  FileText, 
  FileSpreadsheet, 
  Database, 
  ChevronDown, 
  ChevronRight, 
  CheckCircle2, 
  Clock, 
  Cpu, 
  Layers, 
  Filter, 
  Maximize2, 
  Copy, 
  Check, 
  Paperclip, 
  Globe, 
  UploadCloud, 
  PanelLeftClose, 
  PanelLeftOpen, 
  RefreshCw, 
  AlertCircle, 
  Plus, 
  X,
  FileCode,
  Search,
  ExternalLink,
  Brain
} from 'lucide-react';
import UploadModal from './UploadModal';
import WebScrapeModal from './WebScrapeModal';

export default function ChatWorkspace({ 
  status, 
  onRefreshStatus, 
  onSelectImage, 
  onLoadDemo, 
  demoLoading 
}) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: '👋 Welcome to **Multi-Modal RAG Platform**!\n\nI can retrieve, analyze, and synthesize answers across **PDF documents**, **Excel spreadsheets**, **live web pages**, **datasets**, **images**, and **timestamped video keyframes**.\n\nUpload a file, scrape a web page, or choose a suggested prompt below to begin.',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [inputQuery, setInputQuery] = useState('');
  const [selectedSource, setSelectedSource] = useState('All Sources (Search Everything)');
  const [loading, setLoading] = useState(false);
  const [expandedSources, setExpandedSources] = useState({});
  const [copiedIndex, setCopiedIndex] = useState(null);

  // Sidebar toggle state
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Modals state
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [webScrapeModalOpen, setWebScrapeModalOpen] = useState(false);
  const [notification, setNotification] = useState(null);

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const showNotification = (msg, type = 'success') => {
    setNotification({ msg, type });
    setTimeout(() => setNotification(null), 4000);
  };

  // Hero prompt cards shown on empty chat
  const heroPromptCards = [
    {
      icon: FileSpreadsheet,
      iconColor: 'text-emerald-400',
      bgColor: 'bg-emerald-500/10 border-emerald-500/20 hover:border-emerald-500/50',
      title: 'Excel Financial Analysis',
      desc: 'Which product had the highest Q2 revenue in the financial report and what was its growth rate?',
      query: 'Which product had the highest Q2 revenue in the financial report and what was its growth rate?'
    },
    {
      icon: FileText,
      iconColor: 'text-blue-400',
      bgColor: 'bg-blue-500/10 border-blue-500/20 hover:border-blue-500/50',
      title: 'AI Guidelines PDF',
      desc: 'What are the recommended chunk sizes and safety protocols in the AI guidelines PDF?',
      query: 'What are the recommended chunk sizes and safety protocols in the AI guidelines PDF?'
    },
    {
      icon: Film,
      iconColor: 'text-purple-400',
      bgColor: 'bg-purple-500/10 border-purple-500/20 hover:border-purple-500/50',
      title: 'Video Keyframe Search',
      desc: 'What occurred at timestamp 00:02 or 00:04 in the product demo clip?',
      query: 'What occurred at timestamp 00:02 or 00:04 in the product demo clip?'
    },
    {
      icon: ImageIcon,
      iconColor: 'text-indigo-400',
      bgColor: 'bg-indigo-500/10 border-indigo-500/20 hover:border-indigo-500/50',
      title: 'Architecture Diagram',
      desc: 'What components and data flows are shown in the system architecture diagram?',
      query: 'What components are shown in the system architecture diagram?'
    }
  ];

  // Compact suggested questions pills for active chat
  const suggestedPills = [
    { label: '📊 Excel Revenue', query: 'Which product had the highest Q2 revenue in the financial report and what was its growth rate?' },
    { label: '📑 PDF Guidelines', query: 'What are the recommended chunk sizes and safety protocols in the AI guidelines PDF?' },
    { label: '🎥 Video Keyframe', query: 'What occurred at timestamp 00:02 or 00:04 in the product demo clip?' },
    { label: '🖼️ Architecture', query: 'What components are shown in the system architecture diagram?' },
    { label: '📁 Dataset Score', query: 'What accuracy did Gemini-2.5 achieve on adversarial robustness?' },
  ];

  // Helper to get file icon
  const getSourceIcon = (name) => {
    if (name.startsWith('All Sources')) return <Layers className="w-4 h-4 text-blue-400 shrink-0" />;
    const lower = name.toLowerCase();
    if (lower.includes('wikipedia') || lower.includes('http')) return <Globe className="w-4 h-4 text-cyan-400 shrink-0" />;
    if (lower.endsWith('.pdf')) return <FileText className="w-4 h-4 text-rose-400 shrink-0" />;
    if (lower.endsWith('.xlsx') || lower.endsWith('.xls') || lower.endsWith('.csv')) return <FileSpreadsheet className="w-4 h-4 text-emerald-400 shrink-0" />;
    if (lower.endsWith('.json') || lower.endsWith('.jsonl')) return <Database className="w-4 h-4 text-amber-400 shrink-0" />;
    if (lower.endsWith('.png') || lower.endsWith('.jpg') || lower.endsWith('.jpeg') || lower.endsWith('.webp')) return <ImageIcon className="w-4 h-4 text-indigo-400 shrink-0" />;
    if (lower.endsWith('.mp4') || lower.endsWith('.avi') || lower.endsWith('.mov')) return <Film className="w-4 h-4 text-purple-400 shrink-0" />;
    return <FileCode className="w-4 h-4 text-slate-400 shrink-0" />;
  };

  // -------------------------------------------------------------
  // Query Execution
  // -------------------------------------------------------------
  const handleSend = async (queryText = inputQuery, forcedSource = null) => {
    const q = queryText.trim();
    if (!q || loading) return;

    const sourceToUse = forcedSource !== null ? forcedSource : selectedSource;

    const userMsg = {
      role: 'user',
      content: q,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMsg]);
    setInputQuery('');
    setLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: q,
          source_filter: sourceToUse,
          chat_history: messages.slice(-6)
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned error: ${response.statusText}`);
      }

      const data = await response.json();

      const assistantMsg = {
        role: 'assistant',
        content: data.answer,
        visual_evidence: data.visual_evidence || [],
        sources: data.sources || [],
        latency_seconds: data.latency_seconds,
        retrieval_latency_ms: data.retrieval_latency_ms,
        llm_latency_ms: data.llm_latency_ms,
        model_used: data.model_used,
        vector_db_used: data.vector_db_used,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: `❌ **Query Failed:** ${err.message}. Please check your pipeline configuration and vector store.`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handlePromptCardClick = (query) => {
    setSelectedSource('All Sources (Search Everything)');
    handleSend(query, 'All Sources (Search Everything)');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const toggleSources = (index) => {
    setExpandedSources(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  const copyToClipboard = (text, index) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const startNewChat = () => {
    setMessages([
      {
        role: 'assistant',
        content: '👋 New conversation started. Ask me anything about your uploaded multi-modal sources!',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
    ]);
  };

  // Upload/Scrape callbacks
  const handleUploadSuccess = (data) => {
    showNotification(`Indexed ${data.indexed_chunks} chunks from ${data.filename}!`);
    setSelectedSource(data.filename);
    setMessages(prev => [
      ...prev,
      {
        role: 'assistant',
        content: `✅ **File Indexed Successfully!** Ingested **${data.filename}** (${data.indexed_chunks} chunks). Document Focus is now locked to this file.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
    ]);
    if (onRefreshStatus) onRefreshStatus();
  };

  const handleScrapeSuccess = (data) => {
    showNotification(`Indexed ${data.indexed_chunks} chunks from ${data.source_label}!`);
    if (data.source_label) {
      setSelectedSource(data.source_label);
    }
    setMessages(prev => [
      ...prev,
      {
        role: 'assistant',
        content: `✅ **Web Page Indexed!** Successfully scraped and vectorized **${data.source_label}** (${data.indexed_chunks} chunks). Document Focus is now set to this article.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
    ]);
    if (onRefreshStatus) onRefreshStatus();
  };

  const allSources = ['All Sources (Search Everything)', ...(status?.sources || [])];
  const isFocusLocked = selectedSource && !selectedSource.startsWith('All Sources');

  return (
    <div className="flex h-[calc(100vh-4rem)] max-w-full overflow-hidden bg-[#0b0f17]">
      
      {/* ═════════════════════════════════════════════════════════════════════ */}
      {/* LEFT SIDEBAR: KNOWLEDGE BASE & WORKSPACE SOURCES                    */}
      {/* ═════════════════════════════════════════════════════════════════════ */}
      {sidebarOpen && (
        <aside className="w-72 md:w-80 border-r border-slate-800/80 bg-slate-950/70 backdrop-blur-xl flex flex-col shrink-0 overflow-hidden select-none">
          
          {/* Top Actions: New Chat & Collapse */}
          <div className="p-3 border-b border-slate-800/80 flex items-center justify-between gap-2">
            <button
              onClick={startNewChat}
              className="flex-1 flex items-center justify-center space-x-2 py-2 px-3 rounded-xl bg-slate-900 hover:bg-slate-800/90 text-slate-200 border border-slate-800 text-xs font-semibold shadow-sm transition-all hover:border-slate-700"
            >
              <Plus className="w-4 h-4 text-blue-400" />
              <span>New Conversation</span>
            </button>
            <button
              onClick={() => setSidebarOpen(false)}
              className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-900 transition-colors"
              title="Collapse sidebar"
            >
              <PanelLeftClose className="w-4 h-4" />
            </button>
          </div>

          {/* Ingestion Hub Buttons: Upload & Web Scrape */}
          <div className="p-3 space-y-2 border-b border-slate-800/80">
            <div className="text-[10px] uppercase font-bold tracking-wider text-slate-500 px-1">
              Add to Knowledge Base
            </div>
            
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => setUploadModalOpen(true)}
                className="flex items-center justify-center space-x-1.5 py-2 px-2.5 rounded-xl bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-xs font-semibold transition-all hover:border-emerald-400 shadow-sm"
              >
                <UploadCloud className="w-3.5 h-3.5" />
                <span>Upload File</span>
              </button>

              <button
                onClick={() => setWebScrapeModalOpen(true)}
                className="flex items-center justify-center space-x-1.5 py-2 px-2.5 rounded-xl bg-blue-500/10 hover:bg-blue-500/20 text-blue-300 border border-blue-500/30 text-xs font-semibold transition-all hover:border-blue-400 shadow-sm"
              >
                <Globe className="w-3.5 h-3.5" />
                <span>Scrape Web</span>
              </button>
            </div>

            {/* 1-Click Sample Demo Button */}
            <button
              onClick={onLoadDemo}
              disabled={demoLoading}
              className="w-full flex items-center justify-center space-x-1.5 py-2 px-3 rounded-xl bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 border border-amber-500/30 text-xs font-semibold transition-all hover:border-amber-400 disabled:opacity-50"
            >
              {demoLoading ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin text-amber-400" />
                  <span>Indexing 5 Modalities...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                  <span>Load Sample Demo (All 5)</span>
                </>
              )}
            </button>
          </div>

          {/* Sources List */}
          <div className="flex-1 flex flex-col min-h-0 p-3 space-y-2">
            <div className="flex items-center justify-between px-1">
              <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500">
                Indexed Sources ({status?.sources?.length || 0})
              </span>
              <span className="text-[11px] font-mono text-slate-400 bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">
                {status?.total_chunks || 0} chunks
              </span>
            </div>

            <div className="flex-1 overflow-y-auto space-y-1.5 pr-1">
              {allSources.map((src, idx) => {
                const isSelected = selectedSource === src;
                const isAll = src.startsWith('All Sources');
                return (
                  <button
                    key={idx}
                    onClick={() => setSelectedSource(src)}
                    className={`w-full text-left p-2.5 rounded-xl text-xs font-medium flex items-center justify-between transition-all group ${
                      isSelected
                        ? 'bg-blue-600/20 border border-blue-500/60 text-white font-semibold shadow-sm'
                        : 'bg-slate-900/60 hover:bg-slate-900 border border-slate-800/60 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <div className="flex items-center space-x-2.5 min-w-0">
                      {getSourceIcon(src)}
                      <span className="truncate">{src}</span>
                    </div>
                    {isSelected && (
                      <Check className="w-3.5 h-3.5 text-blue-400 shrink-0 ml-1.5" />
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Sidebar Footer: Active DB info */}
          <div className="p-3 border-t border-slate-800/80 bg-slate-950 flex items-center justify-between text-xs text-slate-400">
            <div className="flex items-center space-x-1.5">
              <Database className="w-3.5 h-3.5 text-purple-400" />
              <span>{status?.active_db || 'ChromaDB'}</span>
            </div>
            <span className="text-[11px] font-mono text-emerald-400">● Connected</span>
          </div>

        </aside>
      )}

      {/* ═════════════════════════════════════════════════════════════════════ */}
      {/* MAIN WORKSPACE: CONVERSATION & PROMPT COMPOSER                       */}
      {/* ═════════════════════════════════════════════════════════════════════ */}
      <main className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        
        {/* Toast Notification */}
        {notification && (
          <div className="fixed top-16 right-6 z-50 animate-in slide-in-from-top-3">
            <div className="px-4 py-2.5 rounded-xl bg-slate-900 border border-blue-500/50 shadow-2xl text-xs font-semibold text-white flex items-center space-x-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>{notification.msg}</span>
            </div>
          </div>
        )}

        {/* ── Top Header / Workspace Controls ── */}
        <header className="h-14 border-b border-slate-800/80 bg-slate-950/60 px-4 flex items-center justify-between gap-3 shrink-0">
          <div className="flex items-center space-x-3 min-w-0">
            {!sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(true)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
                title="Expand knowledge sidebar"
              >
                <PanelLeftOpen className="w-5 h-5" />
              </button>
            )}

            {/* Document Focus Chip */}
            <div className="flex items-center space-x-1.5 bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-xl text-xs max-w-xs sm:max-w-md truncate">
              <Filter className="w-3.5 h-3.5 text-blue-400 shrink-0" />
              <span className="text-slate-400 text-[11px]">Searching:</span>
              <span className="font-semibold text-slate-200 truncate">
                {selectedSource}
              </span>
              {isFocusLocked && (
                <button
                  onClick={() => setSelectedSource('All Sources (Search Everything)')}
                  className="p-0.5 hover:text-red-400 ml-1 rounded"
                  title="Clear focus (Search All Sources)"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            {/* Pipeline Info */}
            <div className="hidden lg:flex items-center space-x-2 text-xs text-slate-400 font-mono">
              <span>•</span>
              <span>{status?.llm_model || 'gemini-1.5-flash'}</span>
              <span>•</span>
              <span>Top-K: {status?.retrieval_top_k || 5}</span>
            </div>
          </div>

          {/* Right Action: Clear Chat */}
          <div className="flex items-center space-x-2">
            <button
              onClick={startNewChat}
              className="flex items-center space-x-1 px-3 py-1.5 rounded-xl text-xs font-semibold text-slate-400 hover:text-slate-200 bg-slate-900/80 hover:bg-slate-900 border border-slate-800 transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Clear Chat</span>
            </button>
          </div>
        </header>

        {/* ── Chat Messages Container ── */}
        <div className="flex-1 overflow-y-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
          
          {/* Welcome Hero State (When only initial message exists) */}
          {messages.length === 1 && (
            <div className="max-w-3xl mx-auto py-8 text-center space-y-6 animate-in fade-in duration-300">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-purple-600 flex items-center justify-center mx-auto shadow-xl shadow-blue-500/20 ring-1 ring-white/20">
                <Brain className="w-9 h-9 text-white" />
              </div>

              <div>
                <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
                  Multi-Modal Retrieval Engine
                </h1>
                <p className="text-sm text-slate-400 mt-2 max-w-lg mx-auto">
                  Ask questions across documents, financial spreadsheets, live web pages, images, and timestamped video keyframes.
                </p>
              </div>

              {/* 4 Hero Prompt Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-left pt-2">
                {heroPromptCards.map((card, idx) => {
                  const Icon = card.icon;
                  return (
                    <div
                      key={idx}
                      onClick={() => handlePromptCardClick(card.query)}
                      className={`p-4 rounded-2xl border transition-all cursor-pointer shadow-sm ${card.bgColor} group`}
                    >
                      <div className="flex items-center space-x-2 mb-1.5">
                        <Icon className={`w-4 h-4 ${card.iconColor}`} />
                        <span className="text-xs font-bold text-white group-hover:text-blue-300 transition-colors">
                          {card.title}
                        </span>
                      </div>
                      <p className="text-xs text-slate-400 leading-relaxed">
                        {card.desc}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Active Suggested Pills Row (When messages > 1) */}
          {messages.length > 1 && (
            <div className="flex items-center space-x-2 overflow-x-auto pb-1 max-w-4xl mx-auto scrollbar-none">
              <span className="text-[11px] font-semibold text-slate-500 whitespace-nowrap">Suggested:</span>
              {suggestedPills.map((p, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setSelectedSource('All Sources (Search Everything)');
                    handleSend(p.query, 'All Sources (Search Everything)');
                  }}
                  disabled={loading}
                  className="px-2.5 py-1 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs text-slate-300 hover:text-white whitespace-nowrap transition-colors disabled:opacity-50"
                >
                  {p.label}
                </button>
              ))}
            </div>
          )}

          {/* Messages Thread */}
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.map((msg, index) => {
              const isUser = msg.role === 'user';
              return (
                <div
                  key={index}
                  className={`flex items-start space-x-3 ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}
                >
                  {/* Avatar */}
                  <div
                    className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 shadow-md ${
                      isUser
                        ? 'bg-slate-800 text-white border border-slate-700'
                        : 'bg-gradient-to-tr from-blue-600 to-indigo-600 text-white ring-1 ring-white/10'
                    }`}
                  >
                    {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                  </div>

                  {/* Message Bubble */}
                  <div
                    className={`max-w-[85%] sm:max-w-[80%] rounded-2xl p-4 shadow-sm border ${
                      isUser
                        ? 'bg-slate-800/90 text-slate-100 border-slate-700/80 rounded-tr-none'
                        : 'bg-slate-900/90 text-slate-200 border-slate-800/80 rounded-tl-none'
                    }`}
                  >
                    {/* Header */}
                    <div className="flex items-center justify-between text-[11px] text-slate-400 mb-2">
                      <span className="font-semibold text-slate-300">
                        {isUser ? 'You' : 'Multi-Modal Assistant'}
                      </span>
                      <div className="flex items-center space-x-2">
                        <span>{msg.timestamp}</span>
                        {!isUser && (
                          <button
                            onClick={() => copyToClipboard(msg.content, index)}
                            className="p-1 hover:text-white rounded"
                            title="Copy answer"
                          >
                            {copiedIndex === index ? (
                              <Check className="w-3 h-3 text-emerald-400" />
                            ) : (
                              <Copy className="w-3 h-3" />
                            )}
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Content */}
                    <div className="text-sm leading-relaxed whitespace-pre-wrap font-normal">
                      {msg.content}
                    </div>

                    {/* Performance Telemetry */}
                    {!isUser && msg.latency_seconds !== undefined && (
                      <div className="mt-3 pt-2.5 border-t border-slate-800/80 flex flex-wrap items-center gap-2 text-[11px] text-slate-400 font-mono">
                        <span className="flex items-center space-x-1 text-slate-300">
                          <Clock className="w-3 h-3 text-blue-400" />
                          <span>{msg.latency_seconds}s total</span>
                        </span>
                        <span>•</span>
                        <span>Retrieval: {msg.retrieval_latency_ms}ms</span>
                        <span>•</span>
                        <span>LLM: {msg.llm_latency_ms}ms</span>
                        <span>•</span>
                        <span className="text-blue-400">{msg.model_used}</span>
                      </div>
                    )}

                    {/* Visual Evidence / Video Keyframes */}
                    {!isUser && msg.visual_evidence && msg.visual_evidence.length > 0 && (
                      <div className="mt-3 pt-2.5 border-t border-slate-800/80">
                        <div className="flex items-center space-x-1.5 text-xs font-semibold text-slate-300 mb-2">
                          <Film className="w-3.5 h-3.5 text-purple-400" />
                          <span>Attached Visual Evidence ({msg.visual_evidence.length}):</span>
                        </div>
                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
                          {msg.visual_evidence.map((vis, vIdx) => (
                            <div
                              key={vIdx}
                              onClick={() => onSelectImage(vis)}
                              className="group relative cursor-pointer rounded-xl overflow-hidden border border-slate-800 bg-slate-950 hover:border-blue-500/50 transition-all shadow-md"
                            >
                              <img
                                src={vis.url}
                                alt={vis.filename}
                                className="w-full h-28 object-cover group-hover:scale-105 transition-transform duration-200"
                              />
                              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent flex flex-col justify-end p-2">
                                <span className="text-[11px] font-medium text-white truncate">
                                  {vis.filename}
                                </span>
                              </div>
                              <div className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity bg-black/60 p-1 rounded">
                                <Maximize2 className="w-3 h-3 text-white" />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Retrieved Sources Accordion */}
                    {!isUser && msg.sources && msg.sources.length > 0 && (
                      <div className="mt-3 pt-2 border-t border-slate-800/80">
                        <button
                          onClick={() => toggleSources(index)}
                          className="flex items-center justify-between w-full py-1 text-xs font-semibold text-blue-400 hover:text-blue-300 transition-colors"
                        >
                          <span className="flex items-center space-x-1.5">
                            <FileText className="w-3.5 h-3.5" />
                            <span>Retrieved Sources ({msg.sources.length} chunks)</span>
                          </span>
                          {expandedSources[index] ? (
                            <ChevronDown className="w-3.5 h-3.5" />
                          ) : (
                            <ChevronRight className="w-3.5 h-3.5" />
                          )}
                        </button>

                        {expandedSources[index] && (
                          <div className="mt-2 space-y-2 max-h-60 overflow-y-auto pr-1">
                            {msg.sources.map((src, sIdx) => {
                              const meta = src.metadata || {};
                              const modality = meta.modality || 'text';
                              const dist = src.distance !== undefined ? src.distance : null;
                              return (
                                <div
                                  key={sIdx}
                                  className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs space-y-1.5"
                                >
                                  <div className="flex items-center justify-between">
                                    <div className="flex items-center space-x-2">
                                      <span className="px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-blue-500/10 text-blue-300 border border-blue-500/20">
                                        {modality}
                                      </span>
                                      <span className="font-semibold text-slate-200 truncate max-w-[220px]">
                                        {meta.source || 'Unknown Source'}
                                      </span>
                                    </div>
                                    {dist !== null && (
                                      <span className="text-[10px] text-slate-400 font-mono">
                                        Dist: {dist.toFixed(3)}
                                      </span>
                                    )}
                                  </div>
                                  <p className="text-slate-400 font-mono text-[11px] bg-slate-900 p-2 rounded-lg border border-slate-800/60 leading-relaxed">
                                    {src.content}
                                  </p>
                                </div>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {/* Loading Spinner */}
            {loading && (
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center shrink-0 shadow-md">
                  <Bot className="w-4 h-4 text-white animate-pulse" />
                </div>
                <div className="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-none p-4 shadow-sm">
                  <div className="flex items-center space-x-2 text-xs text-slate-400">
                    <span className="inline-block w-2 h-2 rounded-full bg-blue-500 animate-ping" />
                    <span>Searching vector store & synthesizing multi-modal answer...</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* ── Prompt Input Container ── */}
        <div className="p-4 bg-slate-950/80 border-t border-slate-800/80 shrink-0">
          <div className="max-w-4xl mx-auto relative">
            <div className="rounded-2xl border border-slate-700 bg-slate-900/90 shadow-2xl focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500 transition-all">
              
              {/* Textarea */}
              <textarea
                ref={textareaRef}
                rows={2}
                value={inputQuery}
                onChange={(e) => setInputQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask anything about your documents, spreadsheets, images, or videos... (Press Enter to send)"
                disabled={loading}
                className="w-full bg-transparent resize-none px-4 pt-3 pb-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none scrollbar-none disabled:opacity-50"
              />

              {/* Bottom Actions Row */}
              <div className="flex items-center justify-between px-3 py-2 border-t border-slate-800/60 bg-slate-950/40 rounded-b-2xl">
                <div className="flex items-center space-x-2">
                  
                  {/* 📎 Upload File Modal Button */}
                  <button
                    type="button"
                    onClick={() => setUploadModalOpen(true)}
                    disabled={loading}
                    className="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors shadow-sm"
                    title="Upload PDF, Excel, Dataset, Image, or Video"
                  >
                    <Paperclip className="w-3.5 h-3.5 text-emerald-400" />
                    <span>Upload File</span>
                  </button>

                  {/* 🌐 Web Scrape Modal Button */}
                  <button
                    type="button"
                    onClick={() => setWebScrapeModalOpen(true)}
                    disabled={loading}
                    className="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors shadow-sm"
                    title="Scrape and index web pages"
                  >
                    <Globe className="w-3.5 h-3.5 text-blue-400" />
                    <span>Scrape Web</span>
                  </button>

                  {/* Focus Tag if filtered */}
                  {isFocusLocked && (
                    <div className="hidden sm:flex items-center space-x-1 px-2.5 py-1 rounded-lg bg-blue-500/10 border border-blue-500/30 text-blue-300 text-[11px] max-w-[200px] truncate">
                      <span className="truncate">{selectedSource}</span>
                      <button
                        onClick={() => setSelectedSource('All Sources (Search Everything)')}
                        className="hover:text-red-400 ml-1"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  )}
                </div>

                {/* Send Button */}
                <button
                  onClick={() => handleSend()}
                  disabled={!inputQuery.trim() || loading}
                  className={`w-8 h-8 rounded-xl flex items-center justify-center transition-all ${
                    inputQuery.trim() && !loading
                      ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-md shadow-blue-600/30'
                      : 'bg-slate-800 text-slate-500 cursor-not-allowed'
                  }`}
                  title="Send message"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>

            </div>
          </div>
        </div>

      </main>

      {/* ── Modals ── */}
      <UploadModal
        isOpen={uploadModalOpen}
        onClose={() => setUploadModalOpen(false)}
        onUploadComplete={handleUploadSuccess}
      />

      <WebScrapeModal
        isOpen={webScrapeModalOpen}
        onClose={() => setWebScrapeModalOpen(false)}
        onScrapeComplete={handleScrapeSuccess}
      />

    </div>
  );
}
