import React from 'react';
import { X, ZoomIn, Download, ExternalLink } from 'lucide-react';

export default function ImageModal({ image, onClose }) {
  if (!image) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in">
      <div 
        className="relative max-w-4xl w-full bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-800 bg-slate-950/60">
          <div className="flex items-center space-x-2">
            <span className="text-sm font-semibold text-white">Visual Evidence Preview</span>
            <span className="text-xs text-slate-400 bg-slate-800 px-2 py-0.5 rounded">
              {image.filename || 'Evidence File'}
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Image Display */}
        <div className="p-4 flex items-center justify-center bg-black/40 min-h-[350px] max-h-[70vh] overflow-auto">
          <img
            src={image.url}
            alt={image.filename || 'Visual Evidence'}
            className="max-h-[65vh] w-auto object-contain rounded-lg shadow-lg"
          />
        </div>

        {/* Footer info */}
        <div className="px-5 py-3 bg-slate-950 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
          <span className="truncate max-w-md font-mono">{image.path}</span>
          <a
            href={image.url}
            download={image.filename}
            target="_blank"
            rel="noreferrer"
            className="flex items-center space-x-1 text-blue-400 hover:text-blue-300 font-medium"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download</span>
          </a>
        </div>
      </div>
    </div>
  );
}
