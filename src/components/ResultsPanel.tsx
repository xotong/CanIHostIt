'use client';

import React from 'react';
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

export default function ResultsPanel({ fleet, rackPowerBudgetKw, nodePowerKw }: ResultsPanelProps) {
  const hasModels = fleet.modelResults.length > 0;

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
        <h3 className="text-xs font-medium uppercase tracking-wider mb-3" style={{ color: 'var(--color-text-secondary)' }}>
          Per-Model Breakdown
        </h3>
        {fleet.modelResults.map((r) => (
          <ModelBreakdownRow key={r.entryId} r={r} />
        ))}
      </div>
    </div>
  );
}
