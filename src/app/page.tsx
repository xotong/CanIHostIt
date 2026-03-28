'use client';

import React, { useState, useMemo, useEffect, useCallback } from 'react';
import type { GpuSpec, ModelEntry, ModelSpec } from '@/lib/types';
import { calculateFleetTotals } from '@/lib/calculator';
import { loadGpuInventory, saveGpuInventory, generateId } from '@/lib/gpu-database';
import GpuInventory from '@/components/GpuInventory';
import ModelCard from '@/components/ModelCard';
import ResultsPanel from '@/components/ResultsPanel';
import Tooltip from '@/components/Tooltip';

const EMPTY_MODEL: ModelSpec = {
  name: 'New Model',
  totalParams_B: 0,
  layers: 0,
  kvHeads: 0,
  headDim: 0,
};

function createModelEntry(gpuId: string = ''): ModelEntry {
  return {
    id: `model_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    model: { ...EMPTY_MODEL },
    gpuId,
    quantization: 'FP8',
    kvCacheType: 'FP8',
    maxContextTokens: 131072,
    tierAllocationPercent: 100,
    agenticMultiplier: 1.6,
  };
}

export default function Home() {
  // ─── State ──────────────────────────────────────────────
  const [gpuInventory, setGpuInventory] = useState<GpuSpec[]>([]);
  const [modelEntries, setModelEntries] = useState<ModelEntry[]>([]);
  const [rackPowerBudgetKw, setRackPowerBudgetKw] = useState(20);
  const [nodePowerKw, setNodePowerKw] = useState(10);
  const [totalDevelopers, setTotalDevelopers] = useState(250);
  const [peakActivePercent, setPeakActivePercent] = useState(100);
  const [safetyBufferPercent, setSafetyBufferPercent] = useState(10);
  const [gpuLoaded, setGpuLoaded] = useState(false);
  const [activeTab, setActiveTab] = useState<'models' | 'hardware' | 'settings'>('models');

  // ─── Load GPU inventory from localStorage ───────────────
  useEffect(() => {
    const saved = loadGpuInventory();
    setGpuInventory(saved);
    setGpuLoaded(true);
  }, []);

  // ─── Save GPU inventory to localStorage ─────────────────
  useEffect(() => {
    if (gpuLoaded) {
      saveGpuInventory(gpuInventory);
    }
  }, [gpuInventory, gpuLoaded]);

  // ─── GPU CRUD ───────────────────────────────────────────
  const addGpu = useCallback((gpu: GpuSpec) => setGpuInventory((prev) => [...prev, gpu]), []);
  const removeGpu = useCallback((id: string) => setGpuInventory((prev) => prev.filter((g) => g.id !== id)), []);
  const updateGpu = useCallback((gpu: GpuSpec) => setGpuInventory((prev) => prev.map((g) => (g.id === gpu.id ? gpu : g))), []);

  // ─── Model CRUD ─────────────────────────────────────────
  const addModel = useCallback(() => {
    const defaultGpuId = gpuInventory.length > 0 ? gpuInventory[0].id : '';
    setModelEntries((prev) => [...prev, createModelEntry(defaultGpuId)]);
    setActiveTab('models');
  }, [gpuInventory]);

  const removeModel = useCallback((id: string) => setModelEntries((prev) => prev.filter((m) => m.id !== id)), []);
  const updateModel = useCallback((entry: ModelEntry) => setModelEntries((prev) => prev.map((m) => (m.id === entry.id ? entry : m))), []);

  // ─── Calculation ────────────────────────────────────────
  const fleet = useMemo(
    () => calculateFleetTotals(
      modelEntries,
      gpuInventory,
      totalDevelopers,
      peakActivePercent / 100,
      1 + (safetyBufferPercent / 100),
      rackPowerBudgetKw,
      nodePowerKw
    ),
    [modelEntries, gpuInventory, totalDevelopers, peakActivePercent, safetyBufferPercent, rackPowerBudgetKw, nodePowerKw]
  );

  // Build a results map for inline display
  const resultsMap = useMemo(() => {
    const map = new Map<string, (typeof fleet.modelResults)[0]>();
    fleet.modelResults.forEach((r) => map.set(r.entryId, r));
    return map;
  }, [fleet.modelResults]);

  return (
    <div className="relative z-10">
      {/* ─── Header ────────────────────────────────────── */}
      <header className="px-7 py-5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: 'linear-gradient(135deg, var(--color-accent-cyan), var(--color-accent-violet))' }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="4" y="4" width="6" height="6" rx="1"/><rect x="14" y="4" width="6" height="6" rx="1"/>
              <rect x="4" y="14" width="6" height="6" rx="1"/><rect x="14" y="14" width="6" height="6" rx="1"/>
            </svg>
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight gradient-text">CanIHostIt</h1>
            <p className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>AI Infrastructure Capacity Planner</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={addModel}
            className="text-xs px-4 py-2 rounded-lg font-medium transition-all"
            style={{
              background: 'linear-gradient(135deg, oklch(0.78 0.15 195 / 0.15), oklch(0.65 0.2 290 / 0.15))',
              color: 'var(--color-accent-cyan)',
              border: '1px solid oklch(0.78 0.15 195 / 0.2)',
            }}
          >
            + Add Model
          </button>
        </div>
      </header>

      {/* ─── Dashboard Grid ────────────────────────────── */}
      <div className="dashboard-grid">
        {/* Left Panel */}
        <div className="flex flex-col gap-4">
          {/* Tabs */}
          <div className="toggle-group">
            {(['models', 'hardware', 'settings'] as const).map((tab) => (
              <button
                key={tab}
                className={`toggle-option ${activeTab === tab ? 'active' : ''}`}
                onClick={() => setActiveTab(tab)}
              >
                {tab === 'models' ? `Models (${modelEntries.length})` : tab === 'hardware' ? `GPUs (${gpuInventory.length})` : 'Settings'}
              </button>
            ))}
          </div>

          {/* Models Tab */}
          {activeTab === 'models' && (
            <div className="flex flex-col gap-3 models-scroll">
              {modelEntries.length === 0 ? (
                <div className="glass-card p-6 flex flex-col items-center gap-3">
                  <p className="text-sm" style={{ color: 'var(--color-text-tertiary)' }}>No models configured</p>
                  <button
                    onClick={addModel}
                    className="text-xs px-4 py-2 rounded-lg transition-all"
                    style={{
                      background: 'linear-gradient(135deg, oklch(0.78 0.15 195 / 0.15), oklch(0.65 0.2 290 / 0.15))',
                      color: 'var(--color-accent-cyan)',
                      border: '1px solid oklch(0.78 0.15 195 / 0.2)',
                    }}
                  >
                    + Add Your First Model
                  </button>
                </div>
              ) : (
                modelEntries.map((entry, i) => (
                  <ModelCard
                    key={entry.id}
                    entry={entry}
                    gpus={gpuInventory}
                    results={resultsMap.get(entry.id) || null}
                    onUpdate={updateModel}
                    onRemove={removeModel}
                    index={i}
                    peakActiveRate={peakActivePercent / 100}
                    totalDevelopers={totalDevelopers}
                    safetyBuffer={1 + (safetyBufferPercent / 100)}
                  />
                ))
              )}
              {modelEntries.length > 0 && (
                <button
                  onClick={addModel}
                  className="text-xs px-3 py-2.5 rounded-lg transition-all"
                  style={{
                    background: 'oklch(1 0 0 / 0.03)',
                    color: 'var(--color-text-secondary)',
                    border: '1px dashed oklch(1 0 0 / 0.1)',
                  }}
                >
                  + Add Another Model
                </button>
              )}
            </div>
          )}

          {/* Hardware Tab */}
          {activeTab === 'hardware' && (
            <div className="glass-card p-5">
              <h2 className="text-sm font-semibold mb-3" style={{ color: 'var(--color-text-primary)' }}>GPU Inventory</h2>
              <p className="text-xs mb-4" style={{ color: 'var(--color-text-tertiary)' }}>
                Add GPUs to your inventory. Models are assigned to GPUs from this list.
              </p>
              <GpuInventory gpus={gpuInventory} onAdd={addGpu} onRemove={removeGpu} onUpdate={updateGpu} />
            </div>
          )}

          {/* Settings Tab */}
          {activeTab === 'settings' && (
            <div className="glass-card p-5 flex flex-col gap-4">
              <h2 className="text-sm font-semibold" style={{ color: 'var(--color-text-primary)' }}>Infrastructure Settings</h2>

              <div>
                <Tooltip text="Total number of developers/users in your organization or tenant.">
                  <label className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--color-text-secondary)' }}>
                    Overall Total Developers
                  </label>
                </Tooltip>
                <div className="flex items-center gap-2 mt-1">
                  <input
                    type="number"
                    className="glass-input"
                    value={totalDevelopers}
                    onChange={(e) => setTotalDevelopers(Math.max(1, Number(e.target.value)))}
                    min={1}
                    style={{ width: '110px' }}
                  />
                  <span className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>developers</span>
                </div>
              </div>

              <div>
                <Tooltip text="Peak active user rate (%). Peak Active Users = Total Developers × Peak Active %.">
                  <label className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--color-text-secondary)' }}>
                    Overall Peak Active %
                  </label>
                </Tooltip>
                <div className="flex items-center gap-2 mt-1">
                  <input
                    type="number"
                    className="glass-input"
                    value={peakActivePercent}
                    onChange={(e) => setPeakActivePercent(Math.min(100, Math.max(1, Number(e.target.value))))}
                    min={1}
                    max={100}
                    step={1}
                    style={{ width: '92px' }}
                  />
                  <span className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>%</span>
                </div>
              </div>

              <div>
                <Tooltip text="Safety headroom applied to modeled concurrency to absorb short bursts. Final concurrency is multiplied by (1 + buffer %).">
                  <label className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--color-text-secondary)' }}>
                    Safety Buffer %
                  </label>
                </Tooltip>
                <div className="flex items-center gap-2 mt-1">
                  <input
                    type="number"
                    className="glass-input"
                    value={safetyBufferPercent}
                    onChange={(e) => setSafetyBufferPercent(Math.min(100, Math.max(0, Number(e.target.value))))}
                    min={0}
                    max={100}
                    step={1}
                    style={{ width: '92px' }}
                  />
                  <span className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>%</span>
                </div>
              </div>

              <div>
                <Tooltip text="Power capacity per datacenter rack in kilowatts. Determines how many GPU nodes fit per rack.">
                  <label className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--color-text-secondary)' }}>
                    Rack Power Budget
                  </label>
                </Tooltip>
                <div className="flex items-center gap-2 mt-1">
                  <input
                    type="number"
                    className="glass-input"
                    value={rackPowerBudgetKw}
                    onChange={(e) => setRackPowerBudgetKw(Math.max(1, Number(e.target.value)))}
                    min={1}
                    style={{ width: '100px' }}
                  />
                  <span className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>kW per rack</span>
                </div>
              </div>

              <div>
                <Tooltip text="Power consumption per GPU node (server). Used to calculate how many nodes fit within the rack power budget.">
                  <label className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--color-text-secondary)' }}>
                    Node Power Draw
                  </label>
                </Tooltip>
                <div className="flex items-center gap-2 mt-1">
                  <input
                    type="number"
                    className="glass-input"
                    value={nodePowerKw}
                    onChange={(e) => setNodePowerKw(Math.max(0.5, Number(e.target.value)))}
                    min={0.5}
                    step={0.5}
                    style={{ width: '100px' }}
                  />
                  <span className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>kW per node</span>
                </div>
              </div>

              <div className="pt-2" style={{ borderTop: '1px solid oklch(1 0 0 / 0.06)' }}>
                <p className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
                  Peak Active Users: <strong className="gradient-text">{Math.max(1, Math.ceil(totalDevelopers * (peakActivePercent / 100)))}</strong>
                </p>
                <p className="text-xs mt-1" style={{ color: 'var(--color-text-tertiary)' }}>
                  Safety Buffer Multiplier: <strong className="gradient-text">{(1 + (safetyBufferPercent / 100)).toFixed(2)}x</strong>
                </p>
                <p className="text-xs mt-1" style={{ color: 'var(--color-text-tertiary)' }}>
                  Nodes per rack: <strong className="gradient-text">{nodePowerKw > 0 ? Math.floor(rackPowerBudgetKw / nodePowerKw) : '∞'}</strong>
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Right Panel — Results */}
        <ResultsPanel
           fleet={fleet}
           rackPowerBudgetKw={rackPowerBudgetKw}
            nodePowerKw={nodePowerKw}
            totalDevelopers={totalDevelopers}
            peakActivePercent={peakActivePercent}
            safetyBufferPercent={safetyBufferPercent}
          />
      </div>

      {/* ─── Footer ────────────────────────────────────── */}
      <footer className="px-7 py-4 text-center">
        <p className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
          Batch size auto-optimized from leftover VRAM. 20% vLLM framework overhead. GPU utilization defaults to 90%.
        </p>
      </footer>
    </div>
  );
}
