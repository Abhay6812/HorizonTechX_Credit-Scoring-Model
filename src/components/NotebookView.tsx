import { useState } from 'react';
import { Check, Copy, Play, RotateCcw } from 'lucide-react';
import { NOTEBOOK_CELLS, type NotebookCell } from '../data/notebookCells';
import {
  CorrelationHeatmapPlot,
  EdaDistributionsPlot,
  ModelComparisonBarPlot,
  RocCurvesPlot,
} from './ChartsGallery';

interface NotebookViewProps {
  activeCellId: string;
  onSelectCell: (id: string) => void;
  onDownloadNotebook: () => void;
}

function renderCellPlot(cell: NotebookCell) {
  if (!cell.plotKey) return null;
  switch (cell.plotKey) {
    case 'eda_distributions':
      return (
        <div className="space-y-4 mt-3">
          <EdaDistributionsPlot />
          <CorrelationHeatmapPlot />
        </div>
      );
    case 'model_comparison':
      return (
        <div className="mt-3">
          <ModelComparisonBarPlot />
        </div>
      );
    case 'roc_curves':
      return (
        <div className="mt-3">
          <RocCurvesPlot selectedModel="XGBoost" />
        </div>
      );
    default:
      return null;
  }
}

export function NotebookView({
  activeCellId,
  onSelectCell,
  onDownloadNotebook,
}: NotebookViewProps) {
  const [executedIds, setExecutedIds] = useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    NOTEBOOK_CELLS.forEach((c) => {
      if (c.cellType === 'code') initial[c.id] = true;
    });
    return initial;
  });
  const [runningAll, setRunningAll] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const handleCopy = async (id: string, source: string) => {
    try {
      await navigator.clipboard.writeText(source);
      setCopiedId(id);
      window.setTimeout(() => setCopiedId((prev) => (prev === id ? null : prev)), 1600);
    } catch {
      // Clipboard API unavailable in restricted context
    }
  };

  const handleRunCell = (id: string) => {
    onSelectCell(id);
    setExecutedIds((prev) => ({ ...prev, [id]: false }));
    window.setTimeout(() => {
      setExecutedIds((prev) => ({ ...prev, [id]: true }));
    }, 180);
  };

  const handleClearOutputs = () => {
    setExecutedIds({});
  };

  const handleRunAll = () => {
    setRunningAll(true);
    setExecutedIds({});
    const codeCells = NOTEBOOK_CELLS.filter((c) => c.cellType === 'code');
    codeCells.forEach((cell, index) => {
      window.setTimeout(() => {
        setExecutedIds((prev) => ({ ...prev, [cell.id]: true }));
        if (index === codeCells.length - 1) {
          setRunningAll(false);
        }
      }, 110 * (index + 1));
    });
  };

  return (
    <div className="space-y-6">
      {/* Workspace Toolbar */}
      <div className="bg-white border border-slate-200 rounded-md px-4 py-3 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-base sm:text-lg font-bold text-slate-900 text-balance">
            Credit Scoring Model — Jupyter Notebook
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Python 3.11 · Scikit-Learn + XGBoost · Stratified 5-Fold CV · Matplotlib & Seaborn
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={handleRunAll}
            disabled={runningAll}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-white bg-blue-700 hover:bg-blue-800 disabled:opacity-60 rounded transition-colors whitespace-nowrap shrink-0"
          >
            <Play className="w-3.5 h-3.5" aria-hidden="true" />
            <span>{runningAll ? 'Running Notebook...' : 'Run All Cells'}</span>
          </button>
          <button
            type="button"
            onClick={handleClearOutputs}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 hover:bg-slate-50 rounded transition-colors whitespace-nowrap shrink-0"
          >
            <RotateCcw className="w-3.5 h-3.5" aria-hidden="true" />
            <span>Clear Outputs</span>
          </button>
          <button
            type="button"
            onClick={onDownloadNotebook}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded transition-colors whitespace-nowrap shrink-0"
          >
            <span>Open in Jupyter (.ipynb)</span>
          </button>
        </div>
      </div>

      {/* Notebook Cells */}
      <div className="space-y-4">
        {NOTEBOOK_CELLS.map((cell) => {
          const isSelected = activeCellId === cell.id;
          const hasOutput = executedIds[cell.id];

          if (cell.cellType === 'markdown') {
            return (
              <section
                key={cell.id}
                id={cell.id}
                onClick={() => onSelectCell(cell.id)}
                className={`bg-white border rounded-md p-5 sm:p-6 transition-colors ${
                  isSelected ? 'border-blue-600' : 'border-slate-200'
                }`}
              >
                <div className="text-xs font-medium text-slate-500 mb-2">
                  {cell.section} · Markdown Cell
                </div>
                <h2 className="text-xl font-bold text-slate-900 mb-3 text-balance">
                  TASK 1: Credit Scoring & Creditworthiness Prediction Model
                </h2>
                <p className="text-sm text-slate-700 leading-relaxed max-w-3xl mb-4">
                  Predicts whether an applicant falls into <strong className="text-emerald-700">Low Risk</strong>,{' '}
                  <strong className="text-amber-700">Medium Risk</strong>, or{' '}
                  <strong className="text-red-700">High Risk</strong> tiers using income, existing debt,
                  loan history, payment behavior, credit utilization, and account balance features.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-3 border-t border-slate-100 text-xs">
                  <div>
                    <span className="font-semibold text-slate-900 block mb-0.5">Algorithms</span>
                    <span className="text-slate-600">
                      Logistic Regression, Decision Tree, Random Forest, XGBoost
                    </span>
                  </div>
                  <div>
                    <span className="font-semibold text-slate-900 block mb-0.5">Engineering</span>
                    <span className="text-slate-600">
                      DTI Ratio, Balance-to-Debt, Payment Reliability, StandardScaler
                    </span>
                  </div>
                  <div>
                    <span className="font-semibold text-slate-900 block mb-0.5">Evaluation</span>
                    <span className="text-slate-600">
                      Accuracy, Precision, Recall, F1-Score, OvR ROC-AUC
                    </span>
                  </div>
                </div>
              </section>
            );
          }

          return (
            <section
              key={cell.id}
              id={cell.id}
              onClick={() => onSelectCell(cell.id)}
              className={`bg-white border rounded-md overflow-hidden transition-colors ${
                isSelected ? 'border-blue-600' : 'border-slate-200'
              }`}
            >
              {/* Cell Header */}
              <div className="px-4 py-2.5 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-3 min-w-0">
                  <span className="text-xs font-mono text-slate-500 tabular-nums shrink-0">
                    In [{hasOutput ? cell.executionCount : ' '}]:
                  </span>
                  <span className="text-xs font-semibold text-slate-800 truncate">
                    {cell.title}
                  </span>
                </div>

                <div className="flex items-center gap-1.5 shrink-0">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRunCell(cell.id);
                    }}
                    className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-slate-700 bg-white border border-slate-200 hover:bg-slate-100 rounded transition-colors whitespace-nowrap"
                  >
                    <Play className="w-3 h-3 text-blue-700" aria-hidden="true" />
                    <span>Run Cell</span>
                  </button>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleCopy(cell.id, cell.source);
                    }}
                    className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-slate-700 bg-white border border-slate-200 hover:bg-slate-100 rounded transition-colors whitespace-nowrap"
                  >
                    {copiedId === cell.id ? (
                      <>
                        <Check className="w-3 h-3 text-emerald-600" aria-hidden="true" />
                        <span>Copied</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3 h-3 text-slate-500" aria-hidden="true" />
                        <span>Copy</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Python Code Block */}
              <div className="bg-slate-950 text-slate-100 p-4 overflow-x-auto">
                <pre className="text-xs leading-relaxed font-mono">
                  <code>{cell.source}</code>
                </pre>
              </div>

              {/* Cell Execution Output */}
              {hasOutput && (cell.outputText || cell.plotKey) && (
                <div className="border-t border-slate-200 bg-slate-50/70 p-4">
                  <div className="text-[11px] font-mono text-slate-500 mb-2">
                    Out [{cell.executionCount}]:
                  </div>
                  {cell.outputText && (
                    <pre className="text-xs font-mono text-slate-800 bg-white border border-slate-200 rounded p-3 overflow-x-auto leading-relaxed">
                      {cell.outputText}
                    </pre>
                  )}
                  {renderCellPlot(cell)}
                </div>
              )}
            </section>
          );
        })}
      </div>
    </div>
  );
}
