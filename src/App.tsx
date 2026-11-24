/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import { TopBar, type WorkspaceTab } from './components/TopBar';
import { Sidebar } from './components/Sidebar';
import { NotebookView } from './components/NotebookView';
import { VisualizationsView } from './components/VisualizationsView';
import { MetricsTableView } from './components/MetricsTableView';
import { RiskPredictorView } from './components/RiskPredictorView';
import { NOTEBOOK_CELLS } from './data/notebookCells';
import {
  downloadBenchmarkCsv,
  downloadJupyterNotebook,
} from './utils/notebookExporter';

export default function App() {
  const [activeTab, setActiveTab] = useState<WorkspaceTab>('notebook');
  const [activeCellId, setActiveCellId] = useState<string>(NOTEBOOK_CELLS[0].id);

  const handleJumpToCell = (cellId: string) => {
    setActiveTab('notebook');
    setActiveCellId(cellId);
    window.requestAnimationFrame(() => {
      const target = document.getElementById(cellId);
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      <TopBar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        onDownloadNotebook={downloadJupyterNotebook}
      />

      <div className="flex-1 max-w-[1440px] w-full mx-auto flex flex-col lg:flex-row">
        <Sidebar
          activeTab={activeTab}
          activeCellId={activeCellId}
          onJumpToCell={handleJumpToCell}
          onDownloadCsv={downloadBenchmarkCsv}
          onDownloadNotebook={downloadJupyterNotebook}
        />

        <main className="flex-1 min-w-0 p-4 sm:p-6">
          {activeTab === 'notebook' && (
            <NotebookView
              activeCellId={activeCellId}
              onSelectCell={setActiveCellId}
              onDownloadNotebook={downloadJupyterNotebook}
            />
          )}
          {activeTab === 'visualizations' && <VisualizationsView />}
          {activeTab === 'metrics' && (
            <MetricsTableView onDownloadCsv={downloadBenchmarkCsv} />
          )}
          {activeTab === 'predictor' && <RiskPredictorView />}
        </main>
      </div>
    </div>
  );
}
