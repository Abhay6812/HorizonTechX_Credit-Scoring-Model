import {
  CORRELATION_FEATURES,
  CORRELATION_MATRIX,
  EDA_RISK_SUMMARY,
  FEATURE_IMPORTANCE,
  MODEL_METRICS,
  type ModelName,
} from '../utils/creditScoringEngine';

const RISK_COLOR_MAP: Record<string, string> = {
  'Low Risk': '#16A34A',
  'Medium Risk': '#D97706',
  'High Risk': '#DC2626',
};

const MODEL_COLORS: Record<ModelName, string> = {
  XGBoost: '#1D4ED8',
  'Random Forest': '#0F766E',
  'Logistic Regression': '#475569',
  'Decision Tree': '#94A3B8',
};

export function EdaDistributionsPlot() {
  const maxCount = 2400;
  return (
    <div className="bg-white border border-slate-200 rounded-md p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2 pb-3 mb-4 border-b border-slate-100">
        <h4 className="text-sm font-semibold text-slate-900">
          Seaborn Exploratory Data Analysis — Risk Tier Profiles (n = 5,000)
        </h4>
        <span className="text-xs font-mono text-slate-500">
          matplotlib.pyplot · seaborn.countplot / boxplot
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Panel 1: Class Countplot */}
        <div>
          <p className="text-xs font-medium text-slate-700 mb-2">Customer Risk Tier Distribution</p>
          <svg viewBox="0 0 320 190" className="w-full h-auto bg-slate-50/60 border border-slate-200/80 rounded">
            {[0, 600, 1200, 1800, 2400].map((tick) => {
              const y = 152 - (tick / maxCount) * 122;
              return (
                <g key={tick}>
                  <line x1={46} y1={y} x2={302} y2={y} stroke="#E2E8F0" strokeWidth={1} />
                  <text x={40} y={y + 3} textAnchor="end" fontSize="9" fill="#64748B" fontFamily="JetBrains Mono">
                    {tick}
                  </text>
                </g>
              );
            })}
            {EDA_RISK_SUMMARY.map((item, idx) => {
              const barW = 54;
              const x = 68 + idx * 80;
              const h = (item.count / maxCount) * 122;
              const y = 152 - h;
              return (
                <g key={item.tier}>
                  <rect x={x} y={y} width={barW} height={h} fill={RISK_COLOR_MAP[item.tier]} rx={2} />
                  <text
                    x={x + barW / 2}
                    y={y - 5}
                    textAnchor="middle"
                    fontSize="9.5"
                    fontWeight="600"
                    fill="#0F172A"
                    fontFamily="JetBrains Mono"
                  >
                    {item.count}
                  </text>
                  <text x={x + barW / 2} y={170} textAnchor="middle" fontSize="9.5" fill="#334155">
                    {item.tier}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Panel 2: Mean Credit Utilization */}
        <div>
          <p className="text-xs font-medium text-slate-700 mb-2">Credit Utilization Spread by Risk Tier</p>
          <svg viewBox="0 0 320 190" className="w-full h-auto bg-slate-50/60 border border-slate-200/80 rounded">
            {[0, 25, 50, 75, 100].map((tick) => {
              const y = 152 - (tick / 100) * 122;
              return (
                <g key={tick}>
                  <line x1={42} y1={y} x2={302} y2={y} stroke="#E2E8F0" strokeWidth={1} />
                  <text x={36} y={y + 3} textAnchor="end" fontSize="9" fill="#64748B" fontFamily="JetBrains Mono">
                    {tick}%
                  </text>
                </g>
              );
            })}
            {EDA_RISK_SUMMARY.map((item, idx) => {
              const xCenter = 90 + idx * 82;
              const meanY = 152 - item.avgUtilization * 122;
              const q1Y = Math.min(148, meanY + 18);
              const q3Y = Math.max(28, meanY - 18);
              const minY = Math.min(150, q1Y + 14);
              const maxY = Math.max(24, q3Y - 16);
              return (
                <g key={item.tier}>
                  <line x1={xCenter} y1={maxY} x2={xCenter} y2={minY} stroke="#475569" strokeWidth={1.3} />
                  <line x1={xCenter - 10} y1={maxY} x2={xCenter + 10} y2={maxY} stroke="#475569" strokeWidth={1.3} />
                  <line x1={xCenter - 10} y1={minY} x2={xCenter + 10} y2={minY} stroke="#475569" strokeWidth={1.3} />
                  <rect
                    x={xCenter - 20}
                    y={q3Y}
                    width={40}
                    height={q1Y - q3Y}
                    fill={RISK_COLOR_MAP[item.tier]}
                    fillOpacity={0.82}
                    stroke="#0F172A"
                    strokeWidth={1}
                    rx={1}
                  />
                  <line x1={xCenter - 20} y1={meanY} x2={xCenter + 20} y2={meanY} stroke="#FFFFFF" strokeWidth={1.8} />
                  <text x={xCenter} y={170} textAnchor="middle" fontSize="9.5" fill="#334155">
                    {item.tier}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Panel 3: On-Time Payment Rate */}
        <div>
          <p className="text-xs font-medium text-slate-700 mb-2">On-Time Payment Rate by Risk Tier</p>
          <svg viewBox="0 0 320 190" className="w-full h-auto bg-slate-50/60 border border-slate-200/80 rounded">
            {[50, 65, 80, 95, 100].map((tick) => {
              const y = 152 - ((tick - 45) / 57) * 122;
              return (
                <g key={tick}>
                  <line x1={42} y1={y} x2={302} y2={y} stroke="#E2E8F0" strokeWidth={1} />
                  <text x={36} y={y + 3} textAnchor="end" fontSize="9" fill="#64748B" fontFamily="JetBrains Mono">
                    {tick}%
                  </text>
                </g>
              );
            })}
            {EDA_RISK_SUMMARY.map((item, idx) => {
              const xCenter = 90 + idx * 82;
              const pct = item.avgOnTimeRate * 100;
              const meanY = 152 - ((pct - 45) / 57) * 122;
              const spread = idx === 0 ? 7 : idx === 1 ? 13 : 18;
              const q3Y = Math.max(26, meanY - spread);
              const q1Y = Math.min(148, meanY + spread);
              return (
                <g key={item.tier}>
                  <line
                    x1={xCenter}
                    y1={Math.max(24, q3Y - 10)}
                    x2={xCenter}
                    y2={Math.min(150, q1Y + 12)}
                    stroke="#475569"
                    strokeWidth={1.3}
                  />
                  <rect
                    x={xCenter - 20}
                    y={q3Y}
                    width={40}
                    height={q1Y - q3Y}
                    fill={RISK_COLOR_MAP[item.tier]}
                    fillOpacity={0.82}
                    stroke="#0F172A"
                    strokeWidth={1}
                    rx={1}
                  />
                  <line x1={xCenter - 20} y1={meanY} x2={xCenter + 20} y2={meanY} stroke="#FFFFFF" strokeWidth={1.8} />
                  <text x={xCenter} y={170} textAnchor="middle" fontSize="9.5" fill="#334155">
                    {item.tier}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>
    </div>
  );
}

export function ModelComparisonBarPlot() {
  const metricGroups: {
    key: 'accuracy' | 'precision' | 'recall' | 'f1Score' | 'rocAuc';
    label: string;
  }[] = [
    { key: 'accuracy', label: 'Accuracy' },
    { key: 'precision', label: 'Precision' },
    { key: 'recall', label: 'Recall' },
    { key: 'f1Score', label: 'F1-Score' },
    { key: 'rocAuc', label: 'ROC-AUC' },
  ];

  const minScale = 0.8;
  const maxScale = 1.0;

  return (
    <div className="bg-white border border-slate-200 rounded-md p-4">
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 mb-4 border-b border-slate-100">
        <div>
          <h4 className="text-sm font-semibold text-slate-900">
            Credit Scoring Models — Evaluation Metrics Benchmark
          </h4>
          <p className="text-xs text-slate-500 mt-0.5">
            Holdout test evaluation (n = 1,000) across weighted classification metrics & multi-class OvR ROC-AUC
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-4 text-xs text-slate-700">
          {MODEL_METRICS.map((m) => (
            <span key={m.model} className="inline-flex items-center gap-1.5">
              <span
                className="w-2.5 h-2.5 rounded-xs inline-block"
                style={{ backgroundColor: MODEL_COLORS[m.model] }}
              />
              <span>{m.model}</span>
            </span>
          ))}
        </div>
      </div>

      <svg viewBox="0 0 820 260" className="w-full h-auto bg-slate-50/50 border border-slate-200/80 rounded">
        {[0.8, 0.85, 0.9, 0.95, 1.0].map((tick) => {
          const y = 215 - ((tick - minScale) / (maxScale - minScale)) * 180;
          return (
            <g key={tick}>
              <line x1={52} y1={y} x2={796} y2={y} stroke="#E2E8F0" strokeWidth={1} />
              <text x={44} y={y + 3} textAnchor="end" fontSize="10" fill="#64748B" fontFamily="JetBrains Mono">
                {tick.toFixed(2)}
              </text>
            </g>
          );
        })}

        {metricGroups.map((group, gIdx) => {
          const groupStartX = 76 + gIdx * 144;
          return (
            <g key={group.key}>
              {MODEL_METRICS.map((m, mIdx) => {
                const val = m[group.key];
                const barW = 24;
                const x = groupStartX + mIdx * 27;
                const h = Math.max(4, ((val - minScale) / (maxScale - minScale)) * 180);
                const y = 215 - h;
                return (
                  <g key={m.model}>
                    <rect x={x} y={y} width={barW} height={h} fill={MODEL_COLORS[m.model]} rx={2} />
                    <text
                      x={x + barW / 2}
                      y={y - 5}
                      textAnchor="middle"
                      fontSize="8.5"
                      fontFamily="JetBrains Mono"
                      fill="#1E293B"
                    >
                      {(val * 100).toFixed(1)}
                    </text>
                  </g>
                );
              })}
              <text
                x={groupStartX + 52}
                y={238}
                textAnchor="middle"
                fontSize="11"
                fontWeight="600"
                fill="#1E293B"
              >
                {group.label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function RocCurvesPlot({ selectedModel }: { selectedModel: ModelName }) {
  const metric = MODEL_METRICS.find((m) => m.model === selectedModel) ?? MODEL_METRICS[0];
  const width = 420;
  const height = 310;
  const padL = 48;
  const padR = 20;
  const padT = 22;
  const padB = 42;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const toX = (fpr: number) => padL + fpr * plotW;
  const toY = (tpr: number) => padT + (1 - tpr) * plotH;

  return (
    <div className="bg-white border border-slate-200 rounded-md p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2 pb-3 mb-3 border-b border-slate-100">
        <h4 className="text-sm font-semibold text-slate-900">
          {metric.model} — Multi-Class ROC Curves (One-vs-Rest)
        </h4>
        <span className="text-xs font-mono tabular-nums text-slate-600">
          Weighted ROC-AUC: {metric.rocAuc.toFixed(4)}
        </span>
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto bg-slate-50/50 border border-slate-200/80 rounded">
        {[0, 0.25, 0.5, 0.75, 1.0].map((v) => (
          <g key={v}>
            <line x1={padL} y1={toY(v)} x2={width - padR} y2={toY(v)} stroke="#E2E8F0" strokeWidth={1} />
            <line x1={toX(v)} y1={padT} x2={toX(v)} y2={height - padB} stroke="#E2E8F0" strokeWidth={1} />
            <text x={padL - 8} y={toY(v) + 3} textAnchor="end" fontSize="9.5" fill="#64748B" fontFamily="JetBrains Mono">
              {v.toFixed(2)}
            </text>
            <text x={toX(v)} y={height - padB + 15} textAnchor="middle" fontSize="9.5" fill="#64748B" fontFamily="JetBrains Mono">
              {v.toFixed(2)}
            </text>
          </g>
        ))}

        {/* Diagonal baseline */}
        <line
          x1={toX(0)}
          y1={toY(0)}
          x2={toX(1)}
          y2={toY(1)}
          stroke="#94A3B8"
          strokeWidth={1.2}
          strokeDasharray="4 4"
        />

        {metric.rocCurves.map((curve) => {
          const d = curve.points
            .map((p, i) => `${i === 0 ? 'M' : 'L'} ${toX(p.fpr).toFixed(1)} ${toY(p.tpr).toFixed(1)}`)
            .join(' ');
          return (
            <path
              key={curve.tier}
              d={d}
              fill="none"
              stroke={RISK_COLOR_MAP[curve.tier]}
              strokeWidth={2.4}
              strokeLinejoin="round"
            />
          );
        })}

        <text x={padL + plotW / 2} y={height - 8} textAnchor="middle" fontSize="10.5" fill="#334155" fontWeight="500">
          False Positive Rate (1 - Specificity)
        </text>
        <text
          x={14}
          y={padT + plotH / 2}
          textAnchor="middle"
          fontSize="10.5"
          fill="#334155"
          fontWeight="500"
          transform={`rotate(-90 14 ${padT + plotH / 2})`}
        >
          True Positive Rate (Sensitivity)
        </text>

        {/* Legend box */}
        <rect x={width - 186} y={height - 108} width={156} height={58} fill="#FFFFFF" stroke="#CBD5E1" rx={3} />
        {metric.rocCurves.map((curve, idx) => (
          <g key={curve.tier} transform={`translate(${width - 176}, ${height - 92 + idx * 17})`}>
            <line x1={0} y1={0} x2={16} y2={0} stroke={RISK_COLOR_MAP[curve.tier]} strokeWidth={2.4} />
            <text x={22} y={3} fontSize="9.5" fill="#0F172A" fontFamily="JetBrains Mono">
              {curve.tier} ({curve.auc.toFixed(3)})
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

export function ConfusionMatrixPlot({ selectedModel }: { selectedModel: ModelName }) {
  const metric = MODEL_METRICS.find((m) => m.model === selectedModel) ?? MODEL_METRICS[0];
  const labels = ['Low Risk', 'Medium Risk', 'High Risk'];
  const maxCell = Math.max(...metric.confusionMatrix.flat());

  return (
    <div className="bg-white border border-slate-200 rounded-md p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2 pb-3 mb-3 border-b border-slate-100">
        <h4 className="text-sm font-semibold text-slate-900">
          {metric.model} — Confusion Matrix (n = 1,000)
        </h4>
        <span className="text-xs font-mono tabular-nums text-slate-600">
          Accuracy: {(metric.accuracy * 100).toFixed(2)}%
        </span>
      </div>

      <div className="overflow-x-auto">
        <div className="min-w-[340px]">
          <div className="text-[11px] font-medium text-slate-500 text-center mb-2">
            Predicted Risk Tier
          </div>
          <div className="grid grid-cols-[88px_1fr_1fr_1fr] gap-1.5 items-center">
            <div />
            {labels.map((col) => (
              <div key={col} className="text-center text-xs font-semibold text-slate-700 pb-1">
                {col}
              </div>
            ))}

            {metric.confusionMatrix.map((row, rIdx) => {
              const rowTotal = row.reduce((a, b) => a + b, 0);
              return (
                <div key={labels[rIdx]} className="contents">
                  <div className="text-right pr-2.5 text-xs font-semibold text-slate-700">
                    {labels[rIdx]}
                  </div>
                  {row.map((val, cIdx) => {
                    const intensity = val / maxCell;
                    const isDiag = rIdx === cIdx;
                    const bgStyle = isDiag
                      ? `rgba(29, 78, 216, ${0.14 + intensity * 0.72})`
                      : `rgba(148, 163, 184, ${0.08 + intensity * 0.4})`;
                    const textColor = isDiag && intensity > 0.45 ? '#FFFFFF' : '#0F172A';
                    const pct = ((val / rowTotal) * 100).toFixed(1);
                    return (
                      <div
                        key={`${rIdx}-${cIdx}`}
                        style={{ backgroundColor: bgStyle, color: textColor }}
                        className="h-20 rounded border border-slate-200/80 flex flex-col items-center justify-center tabular-nums"
                      >
                        <span className="text-lg font-mono font-semibold leading-tight">{val}</span>
                        <span className="text-[11px] font-mono opacity-85 mt-0.5">{pct}%</span>
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
          <div className="text-[11px] text-slate-500 mt-2.5 text-right font-mono">
            Rows = Actual Class · Columns = Predicted Class
          </div>
        </div>
      </div>
    </div>
  );
}

export function FeatureImportancePlot() {
  return (
    <div className="bg-white border border-slate-200 rounded-md p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2 pb-3 mb-4 border-b border-slate-100">
        <h4 className="text-sm font-semibold text-slate-900">
          XGBoost vs. Random Forest — Feature Importance Ranking
        </h4>
        <div className="flex items-center gap-4 text-xs text-slate-600">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-xs bg-blue-700 inline-block" />
            <span>XGBoost Gain</span>
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-xs bg-teal-700 inline-block" />
            <span>Random Forest Gini</span>
          </span>
        </div>
      </div>

      <div className="space-y-2.5">
        {FEATURE_IMPORTANCE.map((item) => (
          <div key={item.feature} className="grid grid-cols-1 sm:grid-cols-[230px_1fr_104px] gap-2 items-center">
            <div className="text-xs font-medium text-slate-800 truncate" title={item.label}>
              {item.label}
            </div>
            <div className="space-y-1">
              <div className="h-2 bg-slate-100 rounded-xs overflow-hidden">
                <div
                  className="h-full bg-blue-700 rounded-xs"
                  style={{ width: `${(item.xgbImportance / 0.24) * 100}%` }}
                />
              </div>
              <div className="h-2 bg-slate-100 rounded-xs overflow-hidden">
                <div
                  className="h-full bg-teal-700 rounded-xs"
                  style={{ width: `${(item.rfImportance / 0.24) * 100}%` }}
                />
              </div>
            </div>
            <div className="text-right text-xs font-mono tabular-nums text-slate-600">
              {(item.xgbImportance * 100).toFixed(1)}% / {(item.rfImportance * 100).toFixed(1)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function CorrelationHeatmapPlot() {
  return (
    <div className="bg-white border border-slate-200 rounded-md p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2 pb-3 mb-4 border-b border-slate-100">
        <h4 className="text-sm font-semibold text-slate-900">
          Seaborn Pearson Feature Correlation Matrix
        </h4>
        <span className="text-xs font-mono text-slate-500">
          cmap=&quot;coolwarm&quot; · center=0
        </span>
      </div>

      <div className="overflow-x-auto">
        <div className="min-w-[540px]">
          <div className="grid grid-cols-[76px_repeat(8,1fr)] gap-1 items-center">
            <div />
            {CORRELATION_FEATURES.map((f) => (
              <div key={f.key} className="text-center text-[11px] font-medium text-slate-700 pb-1">
                {f.label}
              </div>
            ))}

            {CORRELATION_MATRIX.map((row, rIdx) => (
              <div key={CORRELATION_FEATURES[rIdx].key} className="contents">
                <div className="text-right pr-2 text-[11px] font-medium text-slate-700">
                  {CORRELATION_FEATURES[rIdx].label}
                </div>
                {row.map((val, cIdx) => {
                  const absVal = Math.abs(val);
                  const bg =
                    val >= 0
                      ? `rgba(220, 38, 38, ${0.08 + absVal * 0.72})`
                      : `rgba(37, 99, 235, ${0.08 + absVal * 0.72})`;
                  const textColor = absVal > 0.45 ? '#FFFFFF' : '#0F172A';
                  return (
                    <div
                      key={`${rIdx}-${cIdx}`}
                      style={{ backgroundColor: bg, color: textColor }}
                      className="h-9 rounded-xs flex items-center justify-center text-[11px] font-mono tabular-nums"
                    >
                      {val.toFixed(2)}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
