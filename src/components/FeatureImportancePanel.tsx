import { useState } from 'react';
import type { RiskPredictionResult } from '../utils/creditScoringEngine';

interface FeatureImportancePanelProps {
  prediction: RiskPredictionResult;
}

type ExplainMode = 'local' | 'global';

export function FeatureImportancePanel({ prediction }: FeatureImportancePanelProps) {
  const [mode, setMode] = useState<ExplainMode>('local');
  const { shapExplanation, model, predictedTier } = prediction;
  const { attributions, baseValue, predictedScore, explainerMethod, importanceMetricLabel } =
    shapExplanation;

  const netShift = predictedScore - baseValue;
  const maxAbsShap = Math.max(0.45, ...attributions.map((a) => a.absImpact));

  const topRiskPushers = attributions.filter((a) => a.shapValue > 0.04).slice(0, 2);
  const topRiskReducers = attributions.filter((a) => a.shapValue < -0.04).slice(0, 2);

  const globalSorted = [...attributions].sort((a, b) => b.globalWeight - a.globalWeight);
  const maxGlobalWeight = Math.max(0.25, ...globalSorted.map((a) => a.globalWeight));

  const buildRationaleText = () => {
    const parts: string[] = [];
    if (topRiskReducers.length > 0) {
      parts.push(
        `favorable ${topRiskReducers
          .map((a) => `${a.label.toLowerCase()} (${a.shapValue.toFixed(2)})`)
          .join(' and ')}`
      );
    }
    if (topRiskPushers.length > 0) {
      parts.push(
        `elevated risk from ${topRiskPushers
          .map((a) => `${a.label.toLowerCase()} (+${a.shapValue.toFixed(2)})`)
          .join(' and ')}`
      );
    }
    if (parts.length === 0) {
      return `${model} keeps this applicant near the population baseline (${baseValue.toFixed(2)}), resulting in a ${predictedTier} classification.`;
    }
    return `${model} classifies this applicant as ${predictedTier} (${predictedScore.toFixed(2)} vs. ${baseValue.toFixed(2)} baseline) primarily driven by ${parts.join(', offset by ')}.`;
  };

  return (
    <section
      aria-label="Feature Importance and SHAP Interpretability"
      className="bg-white border border-slate-200 rounded-md p-5 space-y-5"
    >
      {/* Header & Mode Switch */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-100">
        <div>
          <h2 className="text-sm sm:text-base font-bold text-slate-900">
            Feature Importance &amp; SHAP Prediction Explanation
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            {model} · {explainerMethod} · Explains individual contribution from population baseline
          </p>
        </div>

        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md shrink-0">
          <button
            type="button"
            onClick={() => setMode('local')}
            className={`px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap ${
              mode === 'local'
                ? 'bg-white text-slate-900 shadow-xs font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Local SHAP Explanation
          </button>
          <button
            type="button"
            onClick={() => setMode('global')}
            className={`px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap ${
              mode === 'global'
                ? 'bg-white text-slate-900 shadow-xs font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Global {model} Weights
          </button>
        </div>
      </div>

      {/* Additive SHAP Summary Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 text-xs">
        <div className="p-3 bg-slate-50 border border-slate-200/80 rounded">
          <div className="text-slate-500">Population Baseline E[f(x)]</div>
          <div className="font-mono tabular-nums text-base font-semibold text-slate-900 mt-0.5">
            {baseValue.toFixed(3)}
          </div>
        </div>
        <div className="p-3 bg-slate-50 border border-slate-200/80 rounded">
          <div className="text-slate-500">Net Feature Shift (Σφᵢ)</div>
          <div
            className={`font-mono tabular-nums text-base font-semibold mt-0.5 ${
              netShift > 0 ? 'text-red-700' : netShift < 0 ? 'text-emerald-700' : 'text-slate-900'
            }`}
          >
            {netShift >= 0 ? `+${netShift.toFixed(3)}` : netShift.toFixed(3)}
          </div>
        </div>
        <div className="p-3 bg-slate-50 border border-slate-200/80 rounded">
          <div className="text-slate-500">Final Risk Score f(x)</div>
          <div className="font-mono tabular-nums text-base font-semibold text-slate-900 mt-0.5">
            {predictedScore.toFixed(3)}
          </div>
        </div>
        <div className="p-3 bg-slate-50 border border-slate-200/80 rounded">
          <div className="text-slate-500">Decision Thresholds</div>
          <div className="font-mono tabular-nums text-[11px] font-medium text-slate-700 mt-1">
            &le;1.22 Low · &gt;2.38 High
          </div>
        </div>
      </div>

      {/* Natural Language Rationale */}
      <p className="text-xs sm:text-sm text-slate-700 leading-relaxed bg-slate-50/70 border border-slate-200/70 rounded px-3.5 py-2.5">
        {buildRationaleText()}
      </p>

      {mode === 'local' ? (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500 pb-1">
            <span>Features ranked by absolute local SHAP attribution |φᵢ|</span>
            <div className="flex items-center gap-4">
              <span className="inline-flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-xs bg-emerald-600 inline-block" />
                <span>Lowers Risk (-φ)</span>
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-xs bg-red-600 inline-block" />
                <span>Increases Risk (+φ)</span>
              </span>
            </div>
          </div>

          <div className="space-y-2.5">
            {attributions.map((attr) => {
              const widthPct = Math.min(100, (attr.absImpact / maxAbsShap) * 100);
              const isPositive = attr.shapValue >= 0;
              return (
                <div
                  key={attr.feature}
                  className="grid grid-cols-1 md:grid-cols-[240px_1fr_84px] gap-2 md:gap-4 items-center py-1"
                >
                  <div className="min-w-0">
                    <div className="text-xs font-semibold text-slate-800 truncate">
                      {attr.label}
                    </div>
                    <div className="text-[11px] font-mono tabular-nums text-slate-500 truncate">
                      Value: {attr.featureValue}
                    </div>
                  </div>

                  {/* Diverging Centered SHAP Bar */}
                  <div className="grid grid-cols-2 items-center h-6 bg-slate-50 border border-slate-200/80 rounded-xs overflow-hidden relative">
                    {/* Center Zero Axis */}
                    <div
                      aria-hidden="true"
                      className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-400 z-10"
                    />
                    {/* Negative Side (Left of center, reduces risk) */}
                    <div className="flex justify-end items-center h-full pr-0.5">
                      {!isPositive && (
                        <div
                          className="h-3.5 bg-emerald-600 rounded-l-xs transition-all duration-200"
                          style={{ width: `${widthPct}%` }}
                        />
                      )}
                    </div>
                    {/* Positive Side (Right of center, increases risk) */}
                    <div className="flex justify-start items-center h-full pl-0.5">
                      {isPositive && (
                        <div
                          className="h-3.5 bg-red-600 rounded-r-xs transition-all duration-200"
                          style={{ width: `${widthPct}%` }}
                        />
                      )}
                    </div>
                  </div>

                  <div
                    className={`text-right text-xs font-mono tabular-nums font-semibold ${
                      isPositive ? 'text-red-700' : 'text-emerald-700'
                    }`}
                  >
                    {isPositive ? `+${attr.shapValue.toFixed(3)}` : attr.shapValue.toFixed(3)}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500 pb-1">
            <span>
              {model} Global Feature Importance ({importanceMetricLabel}) vs. Local |φᵢ|
            </span>
            <div className="flex items-center gap-4">
              <span className="inline-flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-xs bg-blue-700 inline-block" />
                <span>Global Model Weight</span>
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-xs bg-slate-500 inline-block" />
                <span>Local |φᵢ| Share</span>
              </span>
            </div>
          </div>

          <div className="space-y-2.5">
            {(() => {
              const totalLocalAbs = Math.max(
                0.001,
                globalSorted.reduce((acc, item) => acc + item.absImpact, 0)
              );
              return globalSorted.map((attr) => {
                const localShare = attr.absImpact / totalLocalAbs;
                return (
                  <div
                    key={attr.feature}
                    className="grid grid-cols-1 md:grid-cols-[220px_1fr_116px] gap-2 md:gap-4 items-center py-0.5"
                  >
                    <div className="text-xs font-medium text-slate-800 truncate">
                      {attr.label}
                    </div>
                    <div className="space-y-1">
                      <div className="h-2 bg-slate-100 rounded-xs overflow-hidden">
                        <div
                          className="h-full bg-blue-700 rounded-xs transition-all duration-200"
                          style={{ width: `${(attr.globalWeight / maxGlobalWeight) * 100}%` }}
                        />
                      </div>
                      <div className="h-1.5 bg-slate-100 rounded-xs overflow-hidden">
                        <div
                          className="h-full bg-slate-500 rounded-xs transition-all duration-200"
                          style={{ width: `${Math.min(100, (localShare / maxGlobalWeight) * 100)}%` }}
                        />
                      </div>
                    </div>
                    <div className="text-right text-xs font-mono tabular-nums text-slate-700">
                      {(attr.globalWeight * 100).toFixed(1)}% / {(localShare * 100).toFixed(1)}%
                    </div>
                  </div>
                );
              });
            })()}
          </div>
        </div>
      )}
    </section>
  );
}
