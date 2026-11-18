import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import ChatWorkspace from './components/ChatWorkspace';
import IngestionHub from './components/IngestionHub';
import ConfigModules from './components/ConfigModules';
import ComparisonLab from './components/ComparisonLab';
import EvaluationDashboard from './components/EvaluationDashboard';
import ImageModal from './components/ImageModal';
import ApiKeyModal from './components/ApiKeyModal';

export default function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [status, setStatus] = useState(null);
  const [demoLoading, setDemoLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [apiKeyModalOpen, setApiKeyModalOpen] = useState(false);
  const [globalToast, setGlobalToast] = useState(null);

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (err) {
      console.warn('API status check offline or connecting...', err);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  const showToast = (message, type = 'success') => {
    setGlobalToast({ message, type });
    setTimeout(() => setGlobalToast(null), 4000);
  };

  const handleLoadDemo = async () => {
    setDemoLoading(true);
    try {
      const res = await fetch('/api/load-sample-data', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Demo data loading failed');
      showToast(data.message, 'success');
      await fetchStatus();
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setDemoLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0b0f17] text-slate-100 flex flex-col font-sans selection:bg-blue-600 selection:text-white">
      {/* Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        status={status}
        onLoadDemo={handleLoadDemo}
        demoLoading={demoLoading}
        onOpenApiKeyModal={() => setApiKeyModalOpen(true)}
      />

      {/* Global Toast */}
      {globalToast && (
        <div className="fixed top-20 right-6 z-50 animate-in slide-in-from-top-4 duration-300">
          <div className={`px-4 py-3 rounded-xl border shadow-2xl text-xs font-semibold flex items-center space-x-2 ${
            globalToast.type === 'success'
              ? 'bg-emerald-950 border-emerald-500/50 text-emerald-200'
              : 'bg-rose-950 border-rose-500/50 text-rose-200'
          }`}>
            <span>{globalToast.message}</span>
          </div>
        </div>
      )}

      {/* Active Tab View */}
      <main className="flex-1">
        {activeTab === 'chat' && (
          <ChatWorkspace
            status={status}
            onRefreshStatus={fetchStatus}
            onSelectImage={setSelectedImage}
            onLoadDemo={handleLoadDemo}
            demoLoading={demoLoading}
          />
        )}

        {activeTab === 'ingest' && (
          <IngestionHub
            status={status}
            onRefreshStatus={fetchStatus}
            onLoadDemo={handleLoadDemo}
            demoLoading={demoLoading}
          />
        )}

        {activeTab === 'config' && (
          <ConfigModules
            status={status}
            onRefreshStatus={fetchStatus}
          />
        )}

        {activeTab === 'compare' && (
          <ComparisonLab
            status={status}
          />
        )}

        {activeTab === 'eval' && (
          <EvaluationDashboard
            status={status}
          />
        )}
      </main>

      {/* Modals */}
      <ImageModal
        image={selectedImage}
        onClose={() => setSelectedImage(null)}
      />

      <ApiKeyModal
        isOpen={apiKeyModalOpen}
        onClose={() => setApiKeyModalOpen(false)}
        onKeySaved={fetchStatus}
      />
    </div>
  );
}
