import React from 'react';
import { 
  Brain, 
  MessageSquare, 
  Sliders, 
  FlaskConical, 
  BarChart3, 
  Sparkles, 
  Database, 
  KeyRound, 
  RefreshCw 
} from 'lucide-react';

export default function Navbar({ 
  activeTab, 
  setActiveTab, 
  status, 
  onLoadDemo, 
  demoLoading, 
  onOpenApiKeyModal 
}) {
  const navItems = [
    { id: 'chat', label: 'Chat & Retrieval', icon: MessageSquare },
    { id: 'compare', label: 'Comparison Lab', icon: FlaskConical },
    { id: 'eval', label: 'Evaluation', icon: BarChart3 },
    { id: 'config', label: 'Pipeline Config', icon: Sliders },
  ];

  return (
    <header className="sticky top-0 z-40 border-b border-slate-800/80 bg-[#0b0f17]/95 backdrop-blur-md select-none">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          
          {/* Logo & Brand */}
          <div 
            className="flex items-center space-x-2.5 cursor-pointer" 
            onClick={() => setActiveTab('chat')}
          >
            <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-purple-600 flex items-center justify-center shadow-lg shadow-blue-500/20 ring-1 ring-white/20">
              <Brain className="w-4 h-4 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-extrabold text-sm sm:text-base tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-300 bg-clip-text text-transparent">
                  Multi-Modal RAG
                </span>
                <span className="px-1.5 py-0.5 text-[9px] font-bold bg-blue-500/10 text-blue-400 border border-blue-500/20 rounded-md">
                  v2.0
                </span>
              </div>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="hidden md:flex items-center space-x-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800/90">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    isActive
                      ? 'bg-blue-600 text-white shadow-md shadow-blue-600/30'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Right Status & Actions */}
          <div className="flex items-center space-x-2.5">
            {/* Quick Demo Button */}
            <button
              onClick={onLoadDemo}
              disabled={demoLoading}
              className="hidden sm:inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 border border-amber-500/30 hover:border-amber-400 transition-all disabled:opacity-50"
              title="Generate & index sample files for all 5 modalities"
            >
              {demoLoading ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin text-amber-400" />
                  <span>Indexing Demo...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                  <span>Load Sample Demo</span>
                </>
              )}
            </button>

            {/* Active Chunks & DB Badge */}
            <div className="flex items-center space-x-1.5 bg-slate-900/90 border border-slate-800 px-2.5 py-1.5 rounded-xl text-xs font-medium text-slate-300">
              <Database className="w-3.5 h-3.5 text-blue-400" />
              <span>{status?.active_db || 'ChromaDB'}:</span>
              <span className="font-bold text-white bg-slate-800 px-1.5 py-0.5 rounded text-[11px]">
                {status?.total_chunks || 0}
              </span>
            </div>

            {/* API Key Modal Button */}
            <button
              onClick={onOpenApiKeyModal}
              className="p-2 rounded-xl bg-slate-900 border border-slate-800 text-slate-400 hover:text-white hover:border-slate-700 transition-colors"
              title="API Key Settings"
            >
              <KeyRound className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Mobile Navigation Tabs */}
        <div className="flex md:hidden overflow-x-auto py-2 space-x-2 border-t border-slate-800/60 scrollbar-none">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                  isActive
                    ? 'bg-blue-600 text-white shadow'
                    : 'text-slate-400 bg-slate-900'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
}
