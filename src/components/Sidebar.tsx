import { Download, FileCode2, TableProperties } from 'lucide-react';
import { NOTEBOOK_CELLS } from '../data/notebookCells';
import type { WorkspaceTab } from './TopBar';

interface SidebarProps {
  activeTab: WorkspaceTab;
  activeCellId: string;
  onJumpToCell: (cellId: string) => void;
  onDownloadCsv: () => void;
  onDownloadNotebook: () => void;
}

export function Sidebar({
  activeTab,
  activeCellId,
  onJumpToCell,
  onDownloadCsv,
  onDownloadNotebook,
}: SidebarProps) {
  return (
    <aside className="w-full lg:w-[264px] shrink-0 border-b lg:border-b-0 lg:border-r border-slate-200 bg-white">
      <div className="p-4 lg:sticky lg:top-16 lg:max-h-[calc(100vh-4rem)] lg:overflow-y-auto space-y-6">
        <div>
          <h2 className="text-xs font-semibold text-slate-500 tracking-wide mb-2">
            Notebook File
          </h2>
          <div className="border border-slate-200 rounded-md p-3 bg-slate-50/70">
            <div className="flex items-center gap-2 text-xs font-mono font-medium text-slate-900">
              <FileCode2 className="w-4 h-4 text-blue-700 shrink-0" aria-hidden="true" />
              <span className="truncate">Credit_Scoring_Model.ipynb</span>
            </div>
            <div className="mt-1.5 text-xs text-slate-500 font-mono tabular-nums">
              Python 3.11 · nbformat 4.5 · 10 cells
            </div>
          </div>
        </div>

        <div>
          <h2 className="text-xs font-semibold text-slate-500 tracking-wide mb-2">
            Pipeline Outline
          </h2>
          <nav aria-label="Notebook Cells" className="space-y-1">
            {NOTEBOOK_CELLS.map((cell, idx) => {
              const isSelected = activeTab === 'notebook' && activeCellId === cell.id;
              return (
                <button
                  key={cell.id}
                  type="button"
                  onClick={() => onJumpToCell(cell.id)}
                  className={`w-full text-left px-2.5 py-2 rounded text-xs transition-colors flex items-start gap-2 ${
                    isSelected
                      ? 'bg-blue-50 text-blue-800 font-semibold'
                      : 'text-slate-700 hover:bg-slate-100'
                  }`}
                >
                  <span className="font-mono text-[11px] text-slate-400 tabular-nums shrink-0 mt-0.5">
                    {String(idx + 1).padStart(2, '0')}
                  </span>
                  <span className="leading-snug">{cell.title}</span>
                </button>
              );
            })}
          </nav>
        </div>

        <div className="pt-4 border-t border-slate-200 space-y-2.5">
          <h2 className="text-xs font-semibold text-slate-500 tracking-wide">
            Project Artifacts
          </h2>
          <dl className="grid grid-cols-2 gap-y-1.5 text-xs">
            <dt className="text-slate-500">Dataset Rows</dt>
            <dd className="text-right font-mono tabular-nums text-slate-900">5,000</dd>
            <dt className="text-slate-500">Features</dt>
            <dd className="text-right font-mono tabular-nums text-slate-900">9 raw / 14 eng</dd>
            <dt className="text-slate-500">Target Classes</dt>
            <dd className="text-right font-mono tabular-nums text-slate-900">3 tiers</dd>
            <dt className="text-slate-500">Top ROC-AUC</dt>
            <dd className="text-right font-mono tabular-nums font-semibold text-blue-700">0.9886</dd>
          </dl>

          <div className="pt-2 space-y-2">
            <button
              type="button"
              onClick={onDownloadNotebook}
              className="w-full inline-flex items-center justify-center gap-2 px-3 py-2 text-xs font-semibold text-white bg-blue-700 hover:bg-blue-800 rounded transition-colors"
            >
              <Download className="w-3.5 h-3.5" aria-hidden="true" />
              <span>Export Jupyter (.ipynb)</span>
            </button>
            <button
              type="button"
              onClick={onDownloadCsv}
              className="w-full inline-flex items-center justify-center gap-2 px-3 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-200 hover:bg-slate-50 rounded transition-colors"
            >
              <TableProperties className="w-3.5 h-3.5" aria-hidden="true" />
              <span>Download CSV Dataset</span>
            </button>
          </div>
        </div>
      </div>
    </aside>
  );
}
