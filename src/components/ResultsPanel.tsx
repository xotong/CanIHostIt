'use client';

import React, { useState } from 'react';
import type { FleetTotals, ModelResults } from '@/lib/types';

interface ResultsPanelProps {
  fleet: FleetTotals;
  rackPowerBudgetKw: number;
  nodePowerKw: number;
}

function formatGiB(value: number): string {
  if (value >= 1024) return `${(value / 1024).toFixed(1)} TiB`;
  if (value >= 100) return `${value.toFixed(0)} GiB`;
  if (value >= 10) return `${value.toFixed(1)} GiB`;
  return `${value.toFixed(2)} GiB`;
}

function MetricCard({ label, value, unit, accentClass, subtitle, stagger }: {
  label: string; value: string; unit?: string; accentClass?: string; subtitle?: string; stagger?: number;
}) {
  return (
    <div className={`metric-card animate-fade-in-up ${accentClass || ''} ${stagger ? `stagger-${stagger}` : ''}`} style={{ opacity: 0 }}>
      <p className="text-xs font-medium uppercase tracking-wider mb-2" style={{ color: 'var(--color-text-tertiary)' }}>{label}</p>
      <div className="flex items-baseline gap-1.5">
        <span className="text-2xl font-bold tracking-tight" style={{ color: 'var(--color-text-primary)' }}>{value}</span>
        {unit && <span className="text-sm font-medium" style={{ color: 'var(--color-text-secondary)' }}>{unit}</span>}
      </div>
      {subtitle && <p className="text-xs mt-1.5" style={{ color: 'var(--color-text-tertiary)' }}>{subtitle}</p>}
    </div>
  );
}

function ModelBreakdownRow({ r }: { r: ModelResults }) {
  return (
    <div className="flex items-center justify-between py-2.5" style={{ borderBottom: '1px solid oklch(1 0 0 / 0.04)' }}>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate" style={{ color: 'var(--color-text-primary)' }}>{r.modelName}</p>
        <p className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>{r.gpuName}</p>
      </div>
      <div className="flex gap-4 text-xs text-right" style={{ color: 'var(--color-text-secondary)' }}>
        <div>
          <p style={{ color: 'var(--color-text-tertiary)' }}>Batch</p>
          <p className="font-medium" style={{ color: 'var(--color-accent-emerald)' }}>{r.effectiveBatchSize}{r.effectiveBatchSize !== r.optimalBatchSize ? '*' : ''}</p>
        </div>
        <div>
          <p style={{ color: 'var(--color-text-tertiary)' }}>TP×PP</p>
          <p className="font-medium" style={{ color: 'var(--color-accent-cyan)' }}>{r.tpSize}×{r.ppSize}</p>
        </div>
        <div>
          <p style={{ color: 'var(--color-text-tertiary)' }}>Replicas</p>
          <p className="font-medium">{r.replicas}</p>
        </div>
        <div>
          <p style={{ color: 'var(--color-text-tertiary)' }}>GPUs</p>
          <p className="font-bold">{r.totalGpus}</p>
        </div>
        <div>
          <p style={{ color: 'var(--color-text-tertiary)' }}>VRAM</p>
          <p className="font-medium">{formatGiB(r.totalVramGiB)}</p>
        </div>
      </div>
    </div>
  );
}

// ─── Calculation Table (detailed view) ────────────────────────────────────────

const TH_STYLE: React.CSSProperties = {
  padding: '8px 12px',
  textAlign: 'left',
  fontSize: '10px',
  fontWeight: 600,
  textTransform: 'uppercase',
  letterSpacing: '0.07em',
  color: 'var(--color-text-tertiary)',
  whiteSpace: 'nowrap',
  borderBottom: '1px solid oklch(1 0 0 / 0.08)',
  background: 'oklch(1 0 0 / 0.02)',
};

const TD_STYLE: React.CSSProperties = {
  padding: '8px 12px',
  fontSize: '12px',
  color: 'var(--color-text-secondary)',
  whiteSpace: 'nowrap',
  borderBottom: '1px solid oklch(1 0 0 / 0.04)',
};

function CalculationTable({ results }: { results: ModelResults[] }) {
  return (
    <div style={{ overflowX: 'auto', borderRadius: '8px', border: '1px solid oklch(1 0 0 / 0.08)' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
        <thead>
          <tr>
            <th style={TH_STYLE}>Model</th>
            <th style={TH_STYLE}>GPU</th>
            <th style={{ ...TH_STYLE, color: 'var(--color-accent-cyan)' }}>Base Weights</th>
            <th style={{ ...TH_STYLE, color: 'var(--color-accent-cyan)' }}>+Overhead (20%)</th>
            <th style={{ ...TH_STYLE, color: 'var(--color-accent-cyan)' }}>Total Weights</th>
            <th style={{ ...TH_STYLE, color: 'oklch(0.75 0.15 290)' }}>Usable VRAM/GPU</th>
            <th style={{ ...TH_STYLE, color: 'oklch(0.75 0.15 290)' }}>TP × PP</th>
            <th style={{ ...TH_STYLE, color: 'oklch(0.75 0.15 290)' }}>GPUs/Replica</th>
            <th style={{ ...TH_STYLE, color: 'var(--color-accent-emerald)' }}>VRAM Left for KV</th>
            <th style={{ ...TH_STYLE, color: 'var(--color-accent-emerald)' }}>KV Cache/User</th>
            <th style={{ ...TH_STYLE, color: 'var(--color-accent-emerald)' }}>Auto Batch</th>
            <th style={{ ...TH_STYLE, color: 'var(--color-accent-emerald)' }}>Eff. Batch</th>
            <th style={{ ...TH_STYLE, color: 'var(--color-accent-amber)' }}>Replicas</th>
            <th style={{ ...TH_STYLE, color: 'var(--color-accent-amber)' }}>Total GPUs</th>
            <th style={{ ...TH_STYLE, color: 'var(--color-accent-amber)' }}>Total Nodes</th>
            <th style={{ ...TH_STYLE, color: 'var(--color-accent-amber)' }}>Total VRAM</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr key={r.entryId} style={{ transition: 'background 0.1s' }}
              onMouseOver={(e) => (e.currentTarget.style.background = 'oklch(1 0 0 / 0.02)')}
              onMouseOut={(e) => (e.currentTarget.style.background = 'transparent')}
            >
              <td style={{ ...TD_STYLE, color: 'var(--color-text-primary)', fontWeight: 600, maxWidth: '160px', overflow: 'hidden', textOverflow: 'ellipsis' }} title={r.modelName}>{r.modelName}</td>
              <td style={{ ...TD_STYLE, color: 'var(--color-text-tertiary)' }}>{r.gpuName}</td>
              {/* Weights */}
              <td style={{ ...TD_STYLE, color: 'var(--color-accent-cyan)' }}>{formatGiB(r.baseWeightsGiB)}</td>
              <td style={{ ...TD_STYLE, color: 'var(--color-accent-cyan)', opacity: 0.75 }}>+{formatGiB(r.frameworkOverheadGiB)}</td>
              <td style={{ ...TD_STYLE, color: 'var(--color-accent-cyan)', fontWeight: 600 }}>{formatGiB(r.totalWeightsGiB)}</td>
              {/* Parallelism */}
              <td style={{ ...TD_STYLE, color: 'oklch(0.75 0.15 290)' }}>{formatGiB(r.usableVramPerGpuGiB)}</td>
              <td style={{ ...TD_STYLE, color: 'oklch(0.75 0.15 290)', fontWeight: 600 }}>{r.tpSize}×{r.ppSize}</td>
              <td style={{ ...TD_STYLE, color: 'oklch(0.75 0.15 290)' }}>{r.gpusPerReplica}</td>
              {/* KV & Batch */}
              <td style={{ ...TD_STYLE, color: 'var(--color-accent-emerald)' }}>{formatGiB(r.vramLeftForKvGiB)}</td>
              <td style={{ ...TD_STYLE, color: 'var(--color-accent-emerald)' }}>{formatGiB(r.kvCachePerUserGiB)}</td>
              <td style={{ ...TD_STYLE, color: 'var(--color-accent-emerald)' }}>{r.optimalBatchSize}</td>
              <td style={{ ...TD_STYLE, color: r.effectiveBatchSize !== r.optimalBatchSize ? 'var(--color-accent-amber)' : 'var(--color-accent-emerald)', fontWeight: 600 }}>
                {r.effectiveBatchSize}{r.effectiveBatchSize !== r.optimalBatchSize ? '*' : ''}
              </td>
              {/* Fleet */}
              <td style={{ ...TD_STYLE, color: 'var(--color-accent-amber)' }}>{r.replicas}</td>
              <td style={{ ...TD_STYLE, color: 'var(--color-accent-amber)', fontWeight: 600 }}>{r.totalGpus}</td>
              <td style={{ ...TD_STYLE, color: 'var(--color-accent-amber)' }}>{r.totalNodes}</td>
              <td style={{ ...TD_STYLE, color: 'var(--color-accent-amber)', fontWeight: 600 }}>{formatGiB(r.totalVramGiB)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs px-3 py-2" style={{ color: 'var(--color-text-tertiary)', borderTop: '1px solid oklch(1 0 0 / 0.06)' }}>
        * Effective batch overridden from auto-optimal. Weights include 20% vLLM framework overhead. Usable VRAM applies GPU utilization cap.
      </p>
    </div>
  );
}

export default function ResultsPanel({ fleet, rackPowerBudgetKw, nodePowerKw }: ResultsPanelProps) {
  const hasModels = fleet.modelResults.length > 0;
  const [showTable, setShowTable] = useState(false);

  if (!hasModels) {
    return (
      <div className="glass-card p-8 flex flex-col items-center justify-center gap-3 min-h-[300px]">
        <div className="w-12 h-12 rounded-2xl flex items-center justify-center" style={{ background: 'oklch(1 0 0 / 0.04)' }}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--color-text-tertiary)" strokeWidth="1.5">
            <rect x="4" y="4" width="6" height="6" rx="1"/><rect x="14" y="4" width="6" height="6" rx="1"/>
            <rect x="4" y="14" width="6" height="6" rx="1"/><rect x="14" y="14" width="6" height="6" rx="1"/>
          </svg>
        </div>
        <p className="text-sm" style={{ color: 'var(--color-text-tertiary)' }}>
          Add models and assign GPUs to see infrastructure requirements
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold tracking-tight" style={{ color: 'var(--color-text-primary)' }}>
          Fleet Requirements
        </h2>
        <p className="text-xs mt-1" style={{ color: 'var(--color-text-tertiary)' }}>
          {fleet.modelResults.length} model{fleet.modelResults.length !== 1 ? 's' : ''} · {rackPowerBudgetKw} kW/rack · {nodePowerKw} kW/node
        </p>
      </div>

      {/* Primary Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricCard label="Total VRAM" value={formatGiB(fleet.totalVramGiB)} accentClass="glow-cyan" subtitle={`${fleet.totalGpus} GPUs total`} stagger={1} />
        <MetricCard label="Total Nodes" value={fleet.totalNodes.toString()} unit="nodes" accentClass="glow-violet" stagger={2} />
        <MetricCard label="Total GPUs" value={fleet.totalGpus.toString()} unit="GPUs" accentClass="glow-emerald" stagger={3} />
        <MetricCard label="Power" value={fleet.totalPowerKw.toFixed(0)} unit="kW" accentClass="glow-amber" subtitle={`${fleet.totalRacks} rack${fleet.totalRacks !== 1 ? 's' : ''}`} stagger={4} />
      </div>

      {/* Per-Model Breakdown */}
      <div className="glass-card p-5 animate-fade-in-up stagger-5" style={{ opacity: 0 }}>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--color-text-secondary)' }}>
            Per-Model Breakdown
          </h3>
          <button
            onClick={() => setShowTable((v) => !v)}
            className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-all"
            style={{
              background: showTable
                ? 'linear-gradient(135deg, oklch(0.78 0.15 195 / 0.2), oklch(0.65 0.2 290 / 0.2))'
                : 'oklch(1 0 0 / 0.04)',
              color: showTable ? 'var(--color-accent-cyan)' : 'var(--color-text-tertiary)',
              border: showTable
                ? '1px solid oklch(0.78 0.15 195 / 0.3)'
                : '1px solid oklch(1 0 0 / 0.06)',
            }}
          >
            {showTable ? (
              <>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>
                  <rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>
                </svg>
                Summary View
              </>
            ) : (
              <>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 3h18v18H3zM3 9h18M3 15h18M9 3v18M15 3v18"/>
                </svg>
                Table View
              </>
            )}
          </button>
        </div>

        {showTable ? (
          <CalculationTable results={fleet.modelResults} />
        ) : (
          fleet.modelResults.map((r) => (
            <ModelBreakdownRow key={r.entryId} r={r} />
          ))
        )}
      </div>
    </div>
  );
}
