import React, { useState } from 'react';
import { 
  UploadCloud, 
  Globe, 
  FileSpreadsheet, 
  FileText, 
  Database, 
  Image as ImageIcon, 
  Film, 
  Sparkles, 
  CheckCircle2, 
  AlertCircle, 
  Trash2, 
  RefreshCw, 
  Layers, 
  ArrowRight,
  Sliders
} from 'lucide-react';

export default function IngestionHub({ 
  status, 
  onRefreshStatus, 
  onLoadDemo, 
  demoLoading 
}) {
  const [activeTab, setActiveTab] = useState('web');
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState(null);

  // Ingestion form states
  const [webUrl, setWebUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [videoFrames, setVideoFrames] = useState(8);
  const [previewImageUrl, setPreviewImageUrl] = useState(null);
  const [confirmWipe, setConfirmWipe] = useState(false);

  const tabs = [
    { id: 'web', label: 'Web Scraper', icon: Globe, color: 'text-blue-400' },
    { id: 'excel', label: 'Excel / CSV', icon: FileSpreadsheet, color: 'text-emerald-400' },
    { id: 'pdf', label: 'PDF Document', icon: FileText, color: 'text-rose-400' },
    { id: 'dataset', label: 'JSON Dataset', icon: Database, color: 'text-amber-400' },
    { id: 'image', label: 'Image', icon: ImageIcon, color: 'text-indigo-400' },
    { id: 'video', label: 'Video Keyframes', icon: Film, color: 'text-purple-400' },
  ];

  const handleFileChange = (e, type) => {
    const file = e.target.files[0];
    if (!file) return;
    setSelectedFile(file);
    if (type === 'image') {
      const url = URL.createObjectURL(file);
      setPreviewImageUrl(url);
    }
  };

  const handleScrapeWeb = async () => {
    if (!webUrl.trim()) return;
    setLoading(true);
    setFeedback(null);
    try {
      const res = await fetch('/api/scrape-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: webUrl })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Scraping failed');
      setFeedback({ type: 'success', message: data.message });
      setWebUrl('');
      onRefreshStatus();
    } catch (err) {
      setFeedback({ type: 'error', message: err.message });
    } finally {
      setLoading(false);
    }
  };

  const handleUploadFile = async (fileType) => {
    if (!selectedFile) return;
    setLoading(true);
    setFeedback(null);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('file_type', fileType);
      formData.append('video_frames', videoFrames.toString());

      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Upload failed');
      setFeedback({ type: 'success', message: data.message });
      setSelectedFile(null);
      setPreviewImageUrl(null);
      onRefreshStatus();
    } catch (err) {
      setFeedback({ type: 'error', message: err.message });
    } finally {
      setLoading(false);
    }
  };

  const handleResetDb = async () => {
    if (!confirmWipe) return;
    setLoading(true);
    setFeedback(null);
    try {
      const res = await fetch('/api/reset-db', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Reset failed');
      setFeedback({ type: 'success', message: data.message });
      setConfirmWipe(false);
      onRefreshStatus();
    } catch (err) {
      setFeedback({ type: 'error', message: err.message });
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
            <UploadCloud className="w-7 h-7 text-blue-500" />
            <span>Data Ingestion Studio</span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Ingest and vectorize multi-modal assets across documents, spreadsheets, live web pages, images, and videos.
          </p>
        </div>

        {/* Total chunks badge */}
        <div className="flex items-center space-x-3 bg-slate-900 border border-slate-800 px-4 py-2 rounded-xl text-xs">
          <Layers className="w-4 h-4 text-emerald-400" />
          <span className="text-slate-400">Total Indexed Chunks:</span>
          <span className="text-base font-bold text-white">{status?.total_chunks || 0}</span>
        </div>
      </div>

      {/* ── Quick Demo Banner ── */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-blue-900/40 via-indigo-900/30 to-purple-900/40 border border-blue-500/30 p-6 shadow-xl">
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="max-w-xl">
            <div className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30 mb-2">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              <span>1-Click Full Modality Test</span>
            </div>
            <h2 className="text-lg font-bold text-white">Load Sample Multi-Modal Demo Data</h2>
            <p className="text-xs text-slate-300 mt-1 leading-relaxed">
              Instantly synthesizes and vectors all 5 modalities: Quarterly Revenue Excel, AI Guidelines PDF, Safety Benchmark JSON, Architecture Diagram PNG, and Demo Video Clip with 6 keyframes.
            </p>
          </div>

          <button
            onClick={onLoadDemo}
            disabled={demoLoading || loading}
            className="flex items-center justify-center space-x-2 px-5 py-3 rounded-xl font-bold text-sm bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-lg shadow-blue-600/30 transition-all disabled:opacity-50 shrink-0"
          >
            {demoLoading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin text-white" />
                <span>Synthesizing & Indexing...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 text-amber-300" />
                <span>Load All 5 Modalities</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </div>
      </div>

      {/* Feedback Toast */}
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

      {/* ── Custom Data Ingestion Tabs ── */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h2 className="text-base font-bold text-white mb-4">Custom Data Ingestion</h2>

        {/* Tab Selection */}
        <div className="flex flex-wrap gap-2 mb-6 border-b border-slate-800/80 pb-3">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => {
                  setActiveTab(tab.id);
                  setSelectedFile(null);
                  setPreviewImageUrl(null);
                  setFeedback(null);
                }}
                className={`flex items-center space-x-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all ${
                  isActive
                    ? 'bg-blue-600 text-white shadow-md shadow-blue-600/30'
                    : 'bg-slate-800/60 text-slate-300 hover:bg-slate-800 hover:text-white'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-white' : tab.color}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Tab 1: Web Scraper */}
        {activeTab === 'web' && (
          <div className="space-y-4 max-w-2xl">
            <label className="block text-xs font-medium text-slate-300">
              Target Web Page URL:
            </label>
            <div className="flex gap-2">
              <input
                type="url"
                value={webUrl}
                onChange={(e) => setWebUrl(e.target.value)}
                placeholder="https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
                className="flex-1 bg-slate-950 border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500"
              />
              <button
                onClick={handleScrapeWeb}
                disabled={loading || !webUrl.trim()}
                className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold rounded-xl shadow-md transition-all disabled:opacity-50"
              >
                {loading ? 'Scraping...' : 'Scrape & Index'}
              </button>
            </div>
            <p className="text-xs text-slate-500">
              Extracts headings, paragraphs, and lists, splitting them into discrete vectors using active chunking settings.
            </p>
          </div>
        )}

        {/* Tab 2: Excel / Spreadsheet */}
        {activeTab === 'excel' && (
          <div className="space-y-4 max-w-2xl">
            <div className="border-2 border-dashed border-slate-700 hover:border-slate-500 rounded-2xl p-6 text-center cursor-pointer transition-colors bg-slate-950/40">
              <FileSpreadsheet className="w-10 h-10 text-emerald-400 mx-auto mb-2" />
              <p className="text-sm font-semibold text-slate-200">
                {selectedFile ? selectedFile.name : 'Upload Excel (.xlsx, .xls) or CSV'}
              </p>
              <p className="text-xs text-slate-500 mt-1">Drag & drop or click to browse</p>
              <input
                type="file"
                accept=".xlsx,.xls,.csv"
                onChange={(e) => handleFileChange(e, 'excel')}
                className="mt-3 text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-500 cursor-pointer"
              />
            </div>
            <button
              onClick={() => handleUploadFile('excel')}
              disabled={loading || !selectedFile}
              className="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded-xl shadow-md transition-all disabled:opacity-50"
            >
              {loading ? 'Processing Spreadsheet...' : 'Index Spreadsheet Rows'}
            </button>
          </div>
        )}

        {/* Tab 3: PDF Document */}
        {activeTab === 'pdf' && (
          <div className="space-y-4 max-w-2xl">
            <div className="border-2 border-dashed border-slate-700 hover:border-slate-500 rounded-2xl p-6 text-center cursor-pointer transition-colors bg-slate-950/40">
              <FileText className="w-10 h-10 text-rose-400 mx-auto mb-2" />
              <p className="text-sm font-semibold text-slate-200">
                {selectedFile ? selectedFile.name : 'Upload PDF Document (.pdf)'}
              </p>
              <p className="text-xs text-slate-500 mt-1">Page text extraction & hierarchical chunking</p>
              <input
                type="file"
                accept=".pdf"
                onChange={(e) => handleFileChange(e, 'pdf')}
                className="mt-3 text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-500 cursor-pointer"
              />
            </div>
            <button
              onClick={() => handleUploadFile('pdf')}
              disabled={loading || !selectedFile}
              className="px-5 py-2.5 bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold rounded-xl shadow-md transition-all disabled:opacity-50"
            >
              {loading ? 'Extracting & Indexing PDF...' : 'Index PDF Document'}
            </button>
          </div>
        )}

        {/* Tab 4: JSON Dataset */}
        {activeTab === 'dataset' && (
          <div className="space-y-4 max-w-2xl">
            <div className="border-2 border-dashed border-slate-700 hover:border-slate-500 rounded-2xl p-6 text-center cursor-pointer transition-colors bg-slate-950/40">
              <Database className="w-10 h-10 text-amber-400 mx-auto mb-2" />
              <p className="text-sm font-semibold text-slate-200">
                {selectedFile ? selectedFile.name : 'Upload JSON or JSONL Dataset'}
              </p>
              <p className="text-xs text-slate-500 mt-1">Record-by-record indexing for benchmark & evaluation datasets</p>
              <input
                type="file"
                accept=".json,.jsonl"
                onChange={(e) => handleFileChange(e, 'dataset')}
                className="mt-3 text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-500 cursor-pointer"
              />
            </div>
            <button
              onClick={() => handleUploadFile('dataset')}
              disabled={loading || !selectedFile}
              className="px-5 py-2.5 bg-amber-600 hover:bg-amber-500 text-white text-xs font-semibold rounded-xl shadow-md transition-all disabled:opacity-50"
            >
              {loading ? 'Indexing Dataset...' : 'Index JSON Dataset'}
            </button>
          </div>
        )}

        {/* Tab 5: Image */}
        {activeTab === 'image' && (
          <div className="space-y-4 max-w-2xl">
            <div className="border-2 border-dashed border-slate-700 hover:border-slate-500 rounded-2xl p-6 text-center cursor-pointer transition-colors bg-slate-950/40">
              <ImageIcon className="w-10 h-10 text-indigo-400 mx-auto mb-2" />
              <p className="text-sm font-semibold text-slate-200">
                {selectedFile ? selectedFile.name : 'Upload Image (.png, .jpg, .webp)'}
              </p>
              <p className="text-xs text-slate-500 mt-1">Multi-modal visual features & OCR description</p>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => handleFileChange(e, 'image')}
                className="mt-3 text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-500 cursor-pointer"
              />
            </div>

            {previewImageUrl && (
              <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 max-w-xs">
                <span className="text-[11px] text-slate-400 font-semibold mb-1 block">Image Preview:</span>
                <img src={previewImageUrl} alt="Preview" className="rounded-lg max-h-48 object-contain" />
              </div>
            )}

            <button
              onClick={() => handleUploadFile('image')}
              disabled={loading || !selectedFile}
              className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-xl shadow-md transition-all disabled:opacity-50"
            >
              {loading ? 'Analyzing & Indexing...' : 'Index Image'}
            </button>
          </div>
        )}

        {/* Tab 6: Video */}
        {activeTab === 'video' && (
          <div className="space-y-4 max-w-2xl">
            <div className="border-2 border-dashed border-slate-700 hover:border-slate-500 rounded-2xl p-6 text-center cursor-pointer transition-colors bg-slate-950/40">
              <Film className="w-10 h-10 text-purple-400 mx-auto mb-2" />
              <p className="text-sm font-semibold text-slate-200">
                {selectedFile ? selectedFile.name : 'Upload Video (.mp4, .avi, .mov)'}
              </p>
              <p className="text-xs text-slate-500 mt-1">Timestamped keyframe extraction & visual captioning</p>
              <input
                type="file"
                accept="video/*"
                onChange={(e) => handleFileChange(e, 'video')}
                className="mt-3 text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-500 cursor-pointer"
              />
            </div>

            {/* Slider for keyframes */}
            <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800">
              <div className="flex justify-between items-center text-xs mb-2">
                <span className="text-slate-300 font-medium">Keyframes to Extract & Caption:</span>
                <span className="font-bold text-purple-400">{videoFrames} frames</span>
              </div>
              <input
                type="range"
                min="4"
                max="16"
                step="2"
                value={videoFrames}
                onChange={(e) => setVideoFrames(parseInt(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
              />
            </div>

            <button
              onClick={() => handleUploadFile('video')}
              disabled={loading || !selectedFile}
              className="px-5 py-2.5 bg-purple-600 hover:bg-purple-500 text-white text-xs font-semibold rounded-xl shadow-md transition-all disabled:opacity-50"
            >
              {loading ? 'Extracting Frames & Indexing...' : 'Extract Keyframes & Index Video'}
            </button>
          </div>
        )}
      </div>

      {/* ── Active Knowledge Base & Wipe Control ── */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
          <div>
            <h2 className="text-base font-bold text-white">Active Knowledge Base Sources</h2>
            <p className="text-xs text-slate-400">Files and URLs currently indexed inside {status?.active_db}</p>
          </div>

          {/* Reset DB Controls */}
          <div className="flex items-center space-x-3">
            <label className="flex items-center space-x-2 text-xs text-rose-300 cursor-pointer">
              <input
                type="checkbox"
                checked={confirmWipe}
                onChange={(e) => setConfirmWipe(e.target.checked)}
                className="rounded border-slate-700 text-rose-600 focus:ring-0"
              />
              <span>Confirm database wipe</span>
            </label>
            <button
              onClick={handleResetDb}
              disabled={!confirmWipe || loading}
              className="flex items-center space-x-1 px-3 py-1.5 text-xs font-semibold bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 rounded-lg transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Clear Vector DB</span>
            </button>
          </div>
        </div>

        {/* Sources Grid */}
        {status?.sources && status.sources.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5">
            {status.sources.map((src, idx) => (
              <div
                key={idx}
                className="flex items-center space-x-2.5 p-3 rounded-xl bg-slate-950/70 border border-slate-800/80 text-xs"
              >
                <div className="w-2 h-2 rounded-full bg-emerald-400 shrink-0" />
                <span className="font-medium text-slate-200 truncate">{src}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-slate-500 text-xs">
            No active sources indexed yet. Click "Load All 5 Modalities" above or upload a custom file!
          </div>
        )}
      </div>

    </div>
  );
}
