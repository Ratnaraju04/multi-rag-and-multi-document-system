import React, { useState } from 'react';
import { 
  X, 
  UploadCloud, 
  FileSpreadsheet, 
  FileText, 
  Database, 
  Image as ImageIcon, 
  Film, 
  CheckCircle2, 
  AlertCircle, 
  RefreshCw,
  File
} from 'lucide-react';

export default function UploadModal({ isOpen, onClose, onUploadComplete }) {
  const [selectedModality, setSelectedModality] = useState('excel');
  const [file, setFile] = useState(null);
  const [videoFrames, setVideoFrames] = useState(8);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  if (!isOpen) return null;

  const modalities = [
    { id: 'excel', label: 'Spreadsheet', icon: FileSpreadsheet, ext: '.xlsx,.xls,.csv', desc: 'Excel & CSV tables with row-by-row chunks' },
    { id: 'pdf', label: 'PDF Document', icon: FileText, ext: '.pdf', desc: 'Sliding window text chunks & page metadata' },
    { id: 'dataset', label: 'JSON Dataset', icon: Database, ext: '.json,.jsonl', desc: 'Structured benchmark records & QA pairs' },
    { id: 'image', label: 'Image Diagram', icon: ImageIcon, ext: 'image/*', desc: 'Visual features, diagram flow & OCR description' },
    { id: 'video', label: 'Video Clip', icon: Film, ext: 'video/*', desc: 'Keyframe extraction with exact timestamps' },
  ];

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      setFile(droppedFile);
      autoDetectModality(droppedFile.name);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      autoDetectModality(selected.name);
    }
  };

  const autoDetectModality = (filename) => {
    const ext = filename.substring(filename.lastIndexOf('.')).toLowerCase();
    if (['.xlsx', '.xls', '.csv'].includes(ext)) setSelectedModality('excel');
    else if (ext === '.pdf') setSelectedModality('pdf');
    else if (['.json', '.jsonl'].includes(ext)) setSelectedModality('dataset');
    else if (['.png', '.jpg', '.jpeg', '.webp', '.bmp'].includes(ext)) setSelectedModality('image');
    else if (['.mp4', '.avi', '.mov', '.mkv'].includes(ext)) setSelectedModality('video');
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('file_type', selectedModality);
      formData.append('video_frames', videoFrames.toString());

      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Upload failed');

      if (onUploadComplete) {
        onUploadComplete(data);
      }
      setFile(null);
      onClose();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const currentModality = modalities.find(m => m.id === selectedModality);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div 
        className="relative max-w-lg w-full bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl p-6 space-y-4"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
              <UploadCloud className="w-4 h-4 text-emerald-400" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white">Upload Multi-Modal Asset</h3>
              <p className="text-xs text-slate-400">Ingest documents, spreadsheets, datasets, images, or videos</p>
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

        {/* Modality Chips */}
        <div>
          <label className="block text-xs font-semibold text-slate-300 mb-2">
            Select Asset Type:
          </label>
          <div className="grid grid-cols-3 sm:grid-cols-5 gap-1.5">
            {modalities.map((m) => {
              const Icon = m.icon;
              const isSelected = selectedModality === m.id;
              return (
                <button
                  key={m.id}
                  onClick={() => setSelectedModality(m.id)}
                  className={`p-2 rounded-xl border text-center transition-all flex flex-col items-center justify-center space-y-1 ${
                    isSelected
                      ? 'bg-blue-600/20 border-blue-500 text-blue-300 ring-1 ring-blue-500 shadow-md'
                      : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="text-[10px] font-semibold">{m.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Drag & Drop Zone */}
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all ${
            dragActive
              ? 'border-blue-500 bg-blue-500/10'
              : 'border-slate-700 hover:border-slate-600 bg-slate-950/50'
          }`}
        >
          <input
            type="file"
            id="modal-file-upload"
            accept={currentModality?.ext || '*'}
            onChange={handleFileSelect}
            className="hidden"
          />
          <label htmlFor="modal-file-upload" className="cursor-pointer block">
            {file ? (
              <div className="space-y-1">
                <File className="w-8 h-8 text-emerald-400 mx-auto mb-2" />
                <span className="text-xs font-bold text-white block truncate max-w-xs mx-auto">
                  {file.name}
                </span>
                <span className="text-[11px] text-slate-400">
                  {(file.size / 1024).toFixed(1)} KB • Click or drop to replace
                </span>
              </div>
            ) : (
              <div className="space-y-1">
                <UploadCloud className="w-8 h-8 text-slate-400 mx-auto mb-2" />
                <span className="text-xs font-semibold text-slate-200 block">
                  Click to select or drag & drop file
                </span>
                <span className="text-[10px] text-slate-500">
                  {currentModality?.desc || 'Any supported format'}
                </span>
              </div>
            )}
          </label>
        </div>

        {/* Video Keyframes Slider */}
        {selectedModality === 'video' && (
          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
            <div className="flex justify-between text-xs text-slate-300 mb-1.5">
              <span className="font-medium">Keyframes to extract & caption:</span>
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
        )}

        {/* Footer Actions */}
        <div className="flex justify-end space-x-2 pt-3 border-t border-slate-800">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-slate-800 text-slate-300 hover:bg-slate-700 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleUpload}
            disabled={loading || !file}
            className="px-5 py-2 rounded-xl text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition-all shadow-md shadow-emerald-600/30 flex items-center space-x-1.5 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Vectorizing File...</span>
              </>
            ) : (
              <>
                <UploadCloud className="w-3.5 h-3.5" />
                <span>Upload & Index File</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
