import { NOTEBOOK_CELLS } from '../data/notebookCells';
import { generateSyntheticRows } from './creditScoringEngine';

export interface JupyterNotebookJSON {
  nbformat: number;
  nbformat_minor: number;
  metadata: {
    kernelspec: {
      display_name: string;
      language: string;
      name: string;
    };
    language_info: {
      name: string;
      version: string;
      mimetype: string;
      codemirror_mode: {
        name: string;
        version: number;
      };
      file_extension: string;
    };
  };
  cells: Array<{
    cell_type: 'markdown' | 'code';
    id: string;
    metadata: Record<string, unknown>;
    source: string[];
    execution_count?: number | null;
    outputs?: Array<Record<string, unknown>>;
  }>;
}

function toNotebookLines(text: string): string[] {
  const lines = text.split('\n');
  return lines.map((line, index) => (index < lines.length - 1 ? `${line}\n` : line));
}

export function buildJupyterNotebookObject(): JupyterNotebookJSON {
  return {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      kernelspec: {
        display_name: 'Python 3 (ipykernel)',
        language: 'python',
        name: 'python3',
      },
      language_info: {
        name: 'python',
        version: '3.11.0',
        mimetype: 'text/x-python',
        codemirror_mode: {
          name: 'ipython',
          version: 3,
        },
        file_extension: '.py',
      },
    },
    cells: NOTEBOOK_CELLS.map((cell) => {
      if (cell.cellType === 'markdown') {
        return {
          cell_type: 'markdown',
          id: cell.id,
          metadata: {},
          source: toNotebookLines(cell.source),
        };
      }

      return {
        cell_type: 'code',
        id: cell.id,
        execution_count: null,
        metadata: {},
        outputs: [],
        source: toNotebookLines(cell.source),
      };
    }),
  };
}

export function downloadJupyterNotebook(): void {
  const notebookObj = buildJupyterNotebookObject();
  const jsonString = JSON.stringify(notebookObj, null, 2);
  const blob = new Blob([jsonString], { type: 'application/x-ipynb+json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'Credit_Scoring_Model.ipynb';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function downloadBenchmarkCsv(): void {
  const rows = generateSyntheticRows(5000);
  const headers = [
    'customer_id',
    'annual_income',
    'existing_debt',
    'loan_history_count',
    'past_defaults',
    'on_time_payment_rate',
    'credit_utilization',
    'account_balance',
    'credit_history_months',
    'monthly_tx_volume',
    'debt_to_income_ratio',
    'risk_category',
  ];

  const csvLines = [
    headers.join(','),
    ...rows.map((r) =>
      [
        r.id,
        r.annual_income,
        r.existing_debt,
        r.loan_history_count,
        r.past_defaults,
        r.on_time_payment_rate,
        r.credit_utilization,
        r.account_balance,
        r.credit_history_months,
        r.monthly_tx_volume,
        r.debt_to_income_ratio,
        `"${r.risk_category}"`,
      ].join(',')
    ),
  ];

  const blob = new Blob([csvLines.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'credit_scoring_dataset.csv';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
