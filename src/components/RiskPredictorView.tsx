import { useMemo, useState } from 'react';
import {
  APPLICANT_PRESETS,
  MODEL_METRICS,
  predictCreditRisk,
  type CustomerFinancialProfile,
  type ModelName,
  type RiskTier,
} from '../utils/creditScoringEngine';
import { FeatureImportancePanel } from './FeatureImportancePanel';

const TIER_STYLES: Record<
  RiskTier,
  { text: string; bar: string; surface: string; summary: string }
> = {
  'Low Risk': {
    text: 'text-emerald-700',
    bar: 'bg-emerald-600',
    surface: 'bg-emerald-50/70 border-emerald-200',
    summary: 'Prime creditworthiness. Eligible for standard or preferred loan terms.',
  },
  'Medium Risk': {
    text: 'text-amber-700',
    bar: 'bg-amber-500',
    surface: 'bg-amber-50/70 border-amber-200',
    summary: 'Moderate credit risk. Recommend manual underwriting review or adjusted rate.',
  },
  'High Risk': {
    text: 'text-red-700',
    bar: 'bg-red-600',
    surface: 'bg-red-50/70 border-red-200',
    summary: 'Elevated default exposure. High probability of delinquency without collateral.',
  },
};

export function RiskPredictorView() {
  const [profile, setProfile] = useState<CustomerFinancialProfile>(
    APPLICANT_PRESETS[0].profile
  );
  const [selectedModel, setSelectedModel] = useState<ModelName>('XGBoost');

  const prediction = useMemo(
    () => predictCreditRisk(profile, selectedModel),
    [profile, selectedModel]
  );

  const crossModelConsensus = useMemo(
    () => MODEL_METRICS.map((m) => predictCreditRisk(profile, m.model)),
    [profile]
  );

  const updateField = <K extends keyof CustomerFinancialProfile>(
    key: K,
    value: number
  ) => {
    setProfile((prev) => ({ ...prev, [key]: value }));
  };

  const tierStyle = TIER_STYLES[prediction.predictedTier];

  return (
    <div className="space-y-6">
      {/* Header & Preset Selector */}
      <div className="bg-white border border-slate-200 rounded-md p-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-base sm:text-lg font-bold text-slate-900">
              Interactive Credit Risk & Creditworthiness Predictor
            </h1>
            <p className="text-xs text-slate-500 mt-0.5">
              Mirrors Cell 8 (`predict_credit_risk`) with real-time engineered ratios and cross-model comparison
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            {APPLICANT_PRESETS.map((preset) => (
              <button
                key={preset.id}
                type="button"
                onClick={() => setProfile(preset.profile)}
                title={preset.summary}
                className="px-2.5 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded transition-colors whitespace-nowrap"
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1.1fr_0.9fr] gap-6 items-start">
        {/* Left: Applicant Financial Inputs */}
        <div className="bg-white border border-slate-200 rounded-md p-5 space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-100">
            <h2 className="text-sm font-semibold text-slate-900">
              Customer Financial & Transaction Inputs
            </h2>
            <button
              type="button"
              onClick={() => setProfile(APPLICANT_PRESETS[0].profile)}
              className="text-xs font-medium text-blue-700 hover:text-blue-800"
            >
              Reset to Default
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Annual Income */}
            <label className="block space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-slate-700">Annual Income</span>
                <span className="font-mono tabular-nums font-semibold text-slate-900">
                  ${profile.annual_income.toLocaleString('en-US')}
                </span>
              </div>
              <input
                type="range"
                min={20000}
                max={240000}
                step={1500}
                value={profile.annual_income}
                onChange={(e) => updateField('annual_income', Number(e.target.value))}
                className="w-full accent-blue-700"
              />
            </label>

            {/* Existing Debt */}
            <label className="block space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-slate-700">Existing Debts</span>
                <span className="font-mono tabular-nums font-semibold text-slate-900">
                  ${profile.existing_debt.toLocaleString('en-US')}
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={160000}
                step={1000}
                value={profile.existing_debt}
                onChange={(e) => updateField('existing_debt', Number(e.target.value))}
                className="w-full accent-blue-700"
              />
            </label>

            {/* Credit Utilization */}
            <label className="block space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-slate-700">Credit Utilization</span>
                <span className="font-mono tabular-nums font-semibold text-slate-900">
                  {(profile.credit_utilization * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min={0.02}
                max={0.98}
                step={0.01}
                value={profile.credit_utilization}
                onChange={(e) => updateField('credit_utilization', Number(e.target.value))}
                className="w-full accent-blue-700"
              />
            </label>

            {/* On-Time Payment Rate */}
            <label className="block space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-slate-700">On-Time Payment Behavior</span>
                <span className="font-mono tabular-nums font-semibold text-slate-900">
                  {(profile.on_time_payment_rate * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min={0.45}
                max={1.0}
                step={0.01}
                value={profile.on_time_payment_rate}
                onChange={(e) => updateField('on_time_payment_rate', Number(e.target.value))}
                className="w-full accent-blue-700"
              />
            </label>

            {/* Account Balance */}
            <label className="block space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-slate-700">Account Balance</span>
                <span className="font-mono tabular-nums font-semibold text-slate-900">
                  ${profile.account_balance.toLocaleString('en-US')}
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={120000}
                step={500}
                value={profile.account_balance}
                onChange={(e) => updateField('account_balance', Number(e.target.value))}
                className="w-full accent-blue-700"
              />
            </label>

            {/* Monthly Transaction Volume */}
            <label className="block space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-slate-700">Monthly Transaction Volume</span>
                <span className="font-mono tabular-nums font-semibold text-slate-900">
                  ${profile.monthly_tx_volume.toLocaleString('en-US')}
                </span>
              </div>
              <input
                type="range"
                min={400}
                max={15000}
                step={150}
                value={profile.monthly_tx_volume}
                onChange={(e) => updateField('monthly_tx_volume', Number(e.target.value))}
                className="w-full accent-blue-700"
              />
            </label>

            {/* Loan History Count */}
            <label className="block space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-slate-700">Historical Loan Accounts</span>
                <span className="font-mono tabular-nums font-semibold text-slate-900">
                  {profile.loan_history_count} loans
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={14}
                step={1}
                value={profile.loan_history_count}
                onChange={(e) => updateField('loan_history_count', Number(e.target.value))}
                className="w-full accent-blue-700"
              />
            </label>

            {/* Past Defaults */}
            <label className="block space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-slate-700">Past Defaults / Charge-offs</span>
                <span className="font-mono tabular-nums font-semibold text-slate-900">
                  {profile.past_defaults}
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={5}
                step={1}
                value={profile.past_defaults}
                onChange={(e) => updateField('past_defaults', Number(e.target.value))}
                className="w-full accent-blue-700"
              />
            </label>

            {/* Credit History Months */}
            <label className="block space-y-1.5 sm:col-span-2">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-slate-700">Credit History Tenure</span>
                <span className="font-mono tabular-nums font-semibold text-slate-900">
                  {profile.credit_history_months} months ({(profile.credit_history_months / 12).toFixed(1)} yrs)
                </span>
              </div>
              <input
                type="range"
                min={6}
                max={264}
                step={6}
                value={profile.credit_history_months}
                onChange={(e) => updateField('credit_history_months', Number(e.target.value))}
                className="w-full accent-blue-700"
              />
            </label>
          </div>

          {/* Engineered Features Readout */}
          <div className="pt-4 border-t border-slate-100">
            <h3 className="text-xs font-semibold text-slate-500 mb-2.5">
              Engineered Feature Transformations (Cell 4)
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-2.5 bg-slate-50 rounded border border-slate-200/70">
                <div className="text-slate-500">Debt-to-Income</div>
                <div className="font-mono tabular-nums font-semibold text-slate-900 mt-0.5">
                  {(prediction.engineeredMetrics.debtToIncomeRatio * 100).toFixed(1)}%
                </div>
              </div>
              <div className="p-2.5 bg-slate-50 rounded border border-slate-200/70">
                <div className="text-slate-500">Balance-to-Debt</div>
                <div className="font-mono tabular-nums font-semibold text-slate-900 mt-0.5">
                  {prediction.engineeredMetrics.balanceToDebtRatio.toFixed(2)}x
                </div>
              </div>
              <div className="p-2.5 bg-slate-50 rounded border border-slate-200/70">
                <div className="text-slate-500">Payment Index</div>
                <div className="font-mono tabular-nums font-semibold text-slate-900 mt-0.5">
                  {prediction.engineeredMetrics.paymentReliabilityIndex.toFixed(3)}
                </div>
              </div>
              <div className="p-2.5 bg-slate-50 rounded border border-slate-200/70">
                <div className="text-slate-500">Util × DTI</div>
                <div className="font-mono tabular-nums font-semibold text-slate-900 mt-0.5">
                  {prediction.engineeredMetrics.utilizationDebtInteraction.toFixed(3)}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Prediction Result & Model Consensus */}
        <div className="space-y-6">
          <div className="bg-white border border-slate-200 rounded-md p-5 space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-100">
              <h2 className="text-sm font-semibold text-slate-900">
                Model Prediction Output
              </h2>
              <div className="flex items-center gap-1 p-0.5 bg-slate-100 rounded">
                {MODEL_METRICS.map((m) => (
                  <button
                    key={m.model}
                    type="button"
                    onClick={() => setSelectedModel(m.model)}
                    className={`px-2 py-1 text-[11px] font-medium rounded transition-colors whitespace-nowrap ${
                      selectedModel === m.model
                        ? 'bg-white text-slate-900 shadow-xs font-semibold'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {m.model}
                  </button>
                ))}
              </div>
            </div>

            <div className={`border rounded-md p-4 ${tierStyle.surface}`}>
              <div className="text-xs font-medium text-slate-600">
                {selectedModel} Classification
              </div>
              <div className={`text-2xl font-bold mt-1 ${tierStyle.text}`}>
                {prediction.predictedTier}
              </div>
              <p className="text-xs text-slate-700 mt-1.5 leading-relaxed">
                {tierStyle.summary}
              </p>
            </div>

            {/* Class Probabilities */}
            <div className="space-y-2.5">
              <div className="text-xs font-semibold text-slate-700">
                Calibrated Class Probabilities
              </div>
              {(['Low Risk', 'Medium Risk', 'High Risk'] as RiskTier[]).map((tier) => {
                const prob = prediction.probabilities[tier];
                return (
                  <div key={tier} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className={`font-medium ${TIER_STYLES[tier].text}`}>{tier}</span>
                      <span className="font-mono tabular-nums text-slate-900">
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2 bg-slate-100 rounded-xs overflow-hidden">
                      <div
                        className={`h-full transition-all duration-200 ${TIER_STYLES[tier].bar}`}
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Primary Factor Breakdown */}
            <div className="pt-3 border-t border-slate-100 space-y-2">
              <div className="text-xs font-semibold text-slate-700">
                Primary Financial Risk Drivers
              </div>
              <div className="divide-y divide-slate-100 text-xs">
                {prediction.drivers.map((driver) => (
                  <div key={driver.label} className="py-2 flex items-center justify-between gap-2">
                    <span className="text-slate-700">{driver.label}</span>
                    <div className="flex items-center gap-3 font-mono tabular-nums">
                      <span className="text-slate-900 font-medium">{driver.valueDisplay}</span>
                      <span
                        className={
                          driver.impact === 'favorable'
                            ? 'text-emerald-700'
                            : driver.impact === 'risk'
                              ? 'text-red-700'
                              : 'text-amber-700'
                        }
                      >
                        {driver.impact === 'favorable'
                          ? 'Favorable'
                          : driver.impact === 'risk'
                            ? 'Elevated Risk'
                            : 'Moderate'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* All 4 Models Consensus Table */}
          <div className="bg-white border border-slate-200 rounded-md p-4">
            <h3 className="text-xs font-semibold text-slate-700 mb-3">
              Cross-Model Consensus Comparison
            </h3>
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2 pr-2 font-semibold">Model</th>
                  <th className="py-2 px-2 font-semibold">Prediction</th>
                  <th className="py-2 pl-2 text-right font-semibold">Confidence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {crossModelConsensus.map((res) => {
                  const isCurrent = res.model === selectedModel;
                  return (
                    <tr
                      key={res.model}
                      onClick={() => setSelectedModel(res.model)}
                      className={`cursor-pointer transition-colors ${
                        isCurrent ? 'bg-blue-50/70' : 'hover:bg-slate-50'
                      }`}
                    >
                      <td className="py-2 pr-2 font-medium text-slate-900">
                        {res.model}
                      </td>
                      <td className={`py-2 px-2 ${TIER_STYLES[res.predictedTier].text}`}>
                        {res.predictedTier}
                      </td>
                      <td className="py-2 pl-2 text-right font-mono tabular-nums text-slate-700">
                        {(res.probabilities[res.predictedTier] * 100).toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* SHAP Feature Importance & Local Interpretability Panel */}
      <FeatureImportancePanel prediction={prediction} />
    </div>
  );
}
