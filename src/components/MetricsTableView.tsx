import { useMemo, useState } from 'react';
import { Download, Search } from 'lucide-react';
import {
  MODEL_METRICS,
  SAMPLE_DATASET_ROWS,
  type ModelName,
  type RiskTier,
} from '../utils/creditScoringEngine';

interface MetricsTableViewProps {
  onDownloadCsv: () => void;
}

const RISK_TEXT_COLORS: Record<RiskTier, string> = {
  'Low Risk': 'text-emerald-700 font-semibold',
  'Medium Risk': 'text-amber-700 font-semibold',
  'High Risk': 'text-red-700 font-semibold',
};

export function MetricsTableView({ onDownloadCsv }: MetricsTableViewProps) {
  const [selectedModel, setSelectedModel] = useState<ModelName>('XGBoost');
  const [riskFilter, setRiskFilter] = useState<'All' | RiskTier>('All');
  const [searchQuery, setSearchQuery] = useState('');

  const activeReport =
    MODEL_METRICS.find((m) => m.model === selectedModel) ?? MODEL_METRICS[0];

  const filteredRows = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return SAMPLE_DATASET_ROWS.filter((row) => {
      if (riskFilter !== 'All' && row.risk_category !== riskFilter) return false;
      if (q && !row.id.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [riskFilter, searchQuery]);

  return (
    <div className="space-y-6">
      {/* Header Summary Strip */}
      <div className="bg-white border border-slate-200 rounded-md p-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2 mb-4">
          <div>
            <h1 className="text-base sm:text-lg font-bold text-slate-900">
              Classification Evaluation Benchmarks
            </h1>
            <p className="text-xs text-slate-500 mt-0.5">
              Stratified 80/20 holdout evaluation (4,000 train / 1,000 test) · 5-fold cross-validation
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500">
            Click any algorithm row to inspect per-class metrics
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-xs font-semibold text-slate-500">
                <th className="py-2.5 pr-4">Algorithm</th>
                <th className="py-2.5 px-3 text-right">Accuracy</th>
                <th className="py-2.5 px-3 text-right">Precision</th>
                <th className="py-2.5 px-3 text-right">Recall</th>
                <th className="py-2.5 px-3 text-right">F1-Score</th>
                <th className="py-2.5 px-3 text-right">ROC-AUC</th>
                <th className="py-2.5 px-3 text-right">5-Fold CV</th>
                <th className="py-2.5 pl-3 text-right">Fit Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-sm">
              {MODEL_METRICS.map((m) => {
                const isSelected = m.model === selectedModel;
                return (
                  <tr
                    key={m.model}
                    onClick={() => setSelectedModel(m.model)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? 'bg-blue-50/70' : 'hover:bg-slate-50'
                    }`}
                  >
                    <td className="py-3 pr-4 font-semibold text-slate-900 whitespace-nowrap">
                      <span>{m.model}</span>
                      {m.model === 'XGBoost' && (
                        <span className="ml-2 text-xs font-normal text-blue-700">
                          · Best OvR AUC
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-800">
                      {(m.accuracy * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-800">
                      {(m.precision * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-800">
                      {(m.recall * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-800">
                      {(m.f1Score * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums font-semibold text-blue-700">
                      {m.rocAuc.toFixed(4)}
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-600">
                      {m.cvRocAuc.toFixed(4)}
                    </td>
                    <td className="py-3 pl-3 text-right font-mono tabular-nums text-slate-500">
                      {m.trainTimeSec.toFixed(2)}s
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Per-Class Classification Report */}
      <div className="bg-white border border-slate-200 rounded-md p-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2 pb-3 mb-3 border-b border-slate-100">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">
              {activeReport.model} — Per-Class Classification Report
            </h2>
            <p className="text-xs font-mono text-slate-500 mt-0.5">
              Best Hyperparameters: {activeReport.bestParams}
            </p>
          </div>
          <span className="text-xs font-mono tabular-nums text-slate-600">
            Support Total: 1,000 test records
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-xs font-semibold text-slate-500">
                <th className="py-2 pr-4">Risk Tier</th>
                <th className="py-2 px-3 text-right">Precision</th>
                <th className="py-2 px-3 text-right">Recall</th>
                <th className="py-2 px-3 text-right">F1-Score</th>
                <th className="py-2 pl-3 text-right">Test Support</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-sm">
              {activeReport.perClassReport.map((row) => (
                <tr key={row.tier} className="hover:bg-slate-50">
                  <td className={`py-2.5 pr-4 ${RISK_TEXT_COLORS[row.tier]}`}>{row.tier}</td>
                  <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-800">
                    {row.precision.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-800">
                    {row.recall.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-800">
                    {row.f1.toFixed(4)}
                  </td>
                  <td className="py-2.5 pl-3 text-right font-mono tabular-nums text-slate-600">
                    {row.support}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Dataset Preview & Filter */}
      <div className="bg-white border border-slate-200 rounded-md p-4">
        <div className="flex flex-wrap items-center justify-between gap-3 pb-3 mb-3 border-b border-slate-100">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">
              Financial & Transaction Dataset Sample (48 of 5,000 records)
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Generated deterministically with seed 42 matching notebook Cell 2
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="relative">
              <Search
                className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-1/2 -translate-y-1/2"
                aria-hidden="true"
              />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Filter Customer ID..."
                aria-label="Filter Customer ID"
                className="pl-8 pr-3 py-1.5 text-xs border border-slate-200 rounded bg-white text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-600"
              />
            </div>

            <div className="flex items-center gap-1 p-0.5 bg-slate-100 rounded">
              {(['All', 'Low Risk', 'Medium Risk', 'High Risk'] as const).map((tier) => (
                <button
                  key={tier}
                  type="button"
                  onClick={() => setRiskFilter(tier)}
                  className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap ${
                    riskFilter === tier
                      ? 'bg-white text-slate-900 shadow-xs font-semibold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {tier}
                </button>
              ))}
            </div>

            <button
              type="button"
              onClick={onDownloadCsv}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 hover:bg-slate-50 rounded transition-colors whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5" aria-hidden="true" />
              <span>Export CSV (5,000)</span>
            </button>
          </div>
        </div>

        {filteredRows.length === 0 ? (
          <div className="py-10 text-center">
            <p className="text-sm font-medium text-slate-800">No matching customer records</p>
            <p className="text-xs text-slate-500 mt-1">
              Clear the Customer ID filter or reset the risk tier selection.
            </p>
            <button
              type="button"
              onClick={() => {
                setSearchQuery('');
                setRiskFilter('All');
              }}
              className="mt-3 px-3 py-1.5 text-xs font-medium text-white bg-slate-900 hover:bg-slate-800 rounded"
            >
              Reset Filters
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto max-h-[380px] overflow-y-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead className="sticky top-0 bg-slate-50 border-b border-slate-200 text-slate-500 font-semibold">
                <tr>
                  <th className="py-2 pr-3">Customer ID</th>
                  <th className="py-2 px-3 text-right">Annual Income</th>
                  <th className="py-2 px-3 text-right">Existing Debt</th>
                  <th className="py-2 px-3 text-right">DTI Ratio</th>
                  <th className="py-2 px-3 text-right">Utilization</th>
                  <th className="py-2 px-3 text-right">On-Time Rate</th>
                  <th className="py-2 px-3 text-right">Defaults</th>
                  <th className="py-2 px-3 text-right">Balance</th>
                  <th className="py-2 pl-3 text-right">Risk Class</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredRows.map((row) => (
                  <tr key={row.id} className="hover:bg-slate-50">
                    <td className="py-2 pr-3 font-mono text-slate-900">{row.id}</td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-800">
                      ${row.annual_income.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-800">
                      ${row.existing_debt.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-700">
                      {(row.debt_to_income_ratio * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-700">
                      {(row.credit_utilization * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-700">
                      {(row.on_time_payment_rate * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-700">
                      {row.past_defaults}
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-800">
                      ${row.account_balance.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                    </td>
                    <td className={`py-2 pl-3 text-right whitespace-nowrap ${RISK_TEXT_COLORS[row.risk_category]}`}>
                      {row.risk_category}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
