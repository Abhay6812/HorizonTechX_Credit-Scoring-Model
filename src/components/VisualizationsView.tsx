import { useState } from 'react';
import { MODEL_METRICS, type ModelName } from '../utils/creditScoringEngine';
import {
  ConfusionMatrixPlot,
  CorrelationHeatmapPlot,
  EdaDistributionsPlot,
  FeatureImportancePlot,
  ModelComparisonBarPlot,
  RocCurvesPlot,
} from './ChartsGallery';

export function VisualizationsView() {
  const [selectedModel, setSelectedModel] = useState<ModelName>('XGBoost');

  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200 rounded-md px-4 py-3 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-base sm:text-lg font-bold text-slate-900">
            Matplotlib & Seaborn Evaluation Visualizations
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Interactive inspection of ROC-AUC curves, confusion matrices, feature correlations, and model benchmarks
          </p>
        </div>

        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md overflow-x-auto">
          {MODEL_METRICS.map((m) => {
            const active = selectedModel === m.model;
            return (
              <button
                key={m.model}
                type="button"
                onClick={() => setSelectedModel(m.model)}
                className={`px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                  active
                    ? 'bg-white text-slate-900 shadow-xs font-semibold'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {m.model}
              </button>
            );
          })}
        </div>
      </div>

      <ModelComparisonBarPlot />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <RocCurvesPlot selectedModel={selectedModel} />
        <ConfusionMatrixPlot selectedModel={selectedModel} />
      </div>

      <FeatureImportancePlot />
      <EdaDistributionsPlot />
      <CorrelationHeatmapPlot />
    </div>
  );
}
