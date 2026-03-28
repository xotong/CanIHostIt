'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import type { ModelSpec, GpuSpec, ModelEntry } from '@/lib/types';
import { searchHuggingFaceModels, fetchFullModelSpec, formatDownloads } from '@/lib/model-database';
import type { HfSearchResult } from '@/lib/types';
import type { ModelResults } from '@/lib/types';
import Tooltip from './Tooltip';

interface ModelCardProps {
  entry: ModelEntry;
  gpus: GpuSpec[];
  results: ModelResults | null;
  onUpdate: (entry: ModelEntry) => void;
  onRemove: (id: string) => void;
  index: number;
}

function formatGiB(value: number): string {
  if (value >= 1024) return `${(value / 1024).toFixed(1)} TiB`;
  if (value >= 100) return `${value.toFixed(0)} GiB`;
  if (value >= 10) return `${value.toFixed(1)} GiB`;
  return `${value.toFixed(2)} GiB`;
}

export default function ModelCard({ entry, gpus, results, onUpdate, onRemove, index }: ModelCardProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<HfSearchResult[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [loading, setLoading] = useState(false);
  const [fetchWarnings, setFetchWarnings] = useState<string[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  const doSearch = useCallback(async (query: string) => {
    if (query.length < 2) { setSearchResults([]); return; }
    const res = await searchHuggingFaceModels(query);
    setSearchResults(res);
    setShowDropdown(res.length > 0);
  }, []);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => doSearch(searchQuery), 400);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [searchQuery, doSearch]);

  const handleSelectModel = async (modelId: string) => {
    setShowDropdown(false);
    setSearchQuery(modelId);
    setLoading(true);
    setFetchWarnings([]);

    const result = await fetchFullModelSpec(modelId);
    setLoading(false);

    if (result) {
      setFetchWarnings(result.warnings);
      const maxCtx = result.spec.maxPositionEmbeddings
        ? Math.min(entry.maxContextTokens, result.spec.maxPositionEmbeddings)
        : entry.maxContextTokens;
      onUpdate({
        ...entry,
        model: result.spec,
        maxContextTokens: maxCtx,
      });
    } else {
      setFetchWarnings(['Failed to fetch model info. Check the model ID or enter details manually.']);
    }
  };

  const updateModel = (patch: Partial<ModelSpec>) => {
    onUpdate({ ...entry, model: { ...entry.model, ...patch } });
  };

  const contextMax = entry.model.maxPositionEmbeddings || 1048576;
  const contextSteps = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576].filter(v => v <= contextMax);
  if (contextSteps.length === 0) contextSteps.push(contextMax);
  const currentCtxIdx = contextSteps.findIndex(v => v >= entry.maxContextTokens);
  const ctxIdx = currentCtxIdx >= 0 ? currentCtxIdx : contextSteps.length - 1;

  const hasBatchOverride = entry.batchSizeOverride !== undefined;

  return (
    <div
      className="glass-card p-5 animate-fade-in-up"
      style={{ opacity: 0, animationDelay: `${index * 0.05}s` }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span
            className="w-6 h-6 rounded-md flex items-center justify-center text-xs font-bold"
            style={{ background: 'linear-gradient(135deg, var(--color-accent-cyan), var(--color-accent-violet))', color: 'white' }}
          >
            {index + 1}
          </span>
          <h3 className="text-sm font-semibold" style={{ color: 'var(--color-text-primary)' }}>
            {entry.model.name || 'New Model'}
          </h3>
        </div>
        <button
          onClick={() => onRemove(entry.id)}
          className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors"
          style={{ background: 'oklch(0.65 0.22 25 / 0.1)', color: 'var(--color-danger)' }}
          title="Remove model"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6 6 18M6 6l12 12"/></svg>
        </button>
      </div>

      {/* HuggingFace Search */}
      <div className="relative mb-3" ref={dropdownRef}>
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <input
              type="text"
              className="glass-input"
              placeholder="Search HuggingFace (e.g. meta-llama, Qwen...)"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onFocus={() => searchResults.length > 0 && setShowDropdown(true)}
              style={{ paddingLeft: '32px' }}
            />
            <svg
              className="absolute left-3 top-1/2 -translate-y-1/2"
              width="14" height="14" viewBox="0 0 24 24" fill="none"
              stroke="var(--color-text-tertiary)" strokeWidth="2"
            >
              <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
            </svg>
          </div>
          {loading && (
            <div className="w-5 h-5 rounded-full border-2 border-t-transparent animate-spin"
              style={{ borderColor: 'var(--color-accent-cyan)', borderTopColor: 'transparent' }}
            />
          )}
        </div>

        {showDropdown && (
          <div
            className="absolute z-50 top-full mt-1 w-full rounded-xl overflow-hidden"
            style={{
              background: 'oklch(0.12 0.01 260 / 0.95)',
              backdropFilter: 'blur(20px)',
              border: '1px solid oklch(1 0 0 / 0.08)',
              boxShadow: '0 12px 40px oklch(0 0 0 / 0.4)',
              maxHeight: '240px',
              overflowY: 'auto',
            }}
          >
            {searchResults.map((r) => (
              <button
                key={r.id}
                className="w-full px-3 py-2.5 text-left transition-colors flex items-center justify-between"
                style={{ borderBottom: '1px solid oklch(1 0 0 / 0.04)' }}
                onClick={() => handleSelectModel(r.id)}
                onMouseOver={(e) => (e.currentTarget.style.background = 'oklch(1 0 0 / 0.04)')}
                onMouseOut={(e) => (e.currentTarget.style.background = 'transparent')}
              >
                <div>
                  <p className="text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>{r.id}</p>
                  <p className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>{r.pipeline_tag} · {r.library_name}</p>
                </div>
                <span className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
                  ↓{formatDownloads(r.downloads)}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Warnings */}
      {fetchWarnings.length > 0 && (
        <div className="mb-3 px-3 py-2 rounded-lg text-xs" style={{ background: 'oklch(0.8 0.16 80 / 0.08)', color: 'var(--color-accent-amber)', border: '1px solid oklch(0.8 0.16 80 / 0.15)' }}>
          {fetchWarnings.map((w, i) => <p key={i}>{w}</p>)}
        </div>
      )}

      {/* Model Info (editable) */}
      <div className="grid grid-cols-4 gap-2 mb-3">
        <div>
          <label className="text-xs mb-0.5 block" style={{ color: 'var(--color-text-tertiary)' }}>Params (B)</label>
          <input type="number" className="glass-input" value={entry.model.totalParams_B || ''} onChange={(e) => updateModel({ totalParams_B: Number(e.target.value) })} min={0.1} step={0.1} />
        </div>
        <div>
          <label className="text-xs mb-0.5 block" style={{ color: 'var(--color-text-tertiary)' }}>Layers</label>
          <input type="number" className="glass-input" value={entry.model.layers || ''} onChange={(e) => updateModel({ layers: Number(e.target.value) })} min={1} />
        </div>
        <div>
          <label className="text-xs mb-0.5 block" style={{ color: 'var(--color-text-tertiary)' }}>KV Heads</label>
          <input type="number" className="glass-input" value={entry.model.kvHeads || ''} onChange={(e) => updateModel({ kvHeads: Number(e.target.value) })} min={1} />
        </div>
        <div>
          <label className="text-xs mb-0.5 block" style={{ color: 'var(--color-text-tertiary)' }}>Head Dim</label>
          <input type="number" className="glass-input" value={entry.model.headDim || ''} onChange={(e) => updateModel({ headDim: Number(e.target.value) })} min={1} />
        </div>
      </div>

      {/* GPU Assignment */}
      <div className="mb-3">
        <label className="text-xs mb-0.5 block" style={{ color: 'var(--color-text-tertiary)' }}>Assigned GPU</label>
        {gpus.length > 0 ? (
          <select
            className="glass-select"
            value={entry.gpuId}
            onChange={(e) => onUpdate({ ...entry, gpuId: e.target.value })}
          >
            <option value="">Select GPU…</option>
            {gpus.map((g) => (
              <option key={g.id} value={g.id}>{g.name} ({g.vramGiB} GiB)</option>
            ))}
          </select>
        ) : (
          <p className="text-xs py-2" style={{ color: 'var(--color-accent-amber)' }}>
            Add GPUs to your inventory first
          </p>
        )}
      </div>

      {/* Quantization + KV Cache Type */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <div>
          <Tooltip text="FP8 uses 1 byte per parameter (faster, less VRAM). BF16 uses 2 bytes (higher precision, more VRAM).">
            <label className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--color-text-secondary)' }}>
              Weight Precision
            </label>
          </Tooltip>
          <div className="toggle-group mt-1">
            {(['FP8', 'BF16'] as const).map((q) => (
              <button key={q} className={`toggle-option ${entry.quantization === q ? 'active' : ''}`} onClick={() => onUpdate({ ...entry, quantization: q })}>
                {q}
              </button>
            ))}
          </div>
        </div>
        <div>
          <Tooltip text="KV cache precision. FP8 (1 byte) halves KV memory vs BF16 (2 bytes), doubling batch size capacity.">
            <label className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--color-text-secondary)' }}>
              KV Cache Type
            </label>
          </Tooltip>
          <div className="toggle-group mt-1">
            {(['FP8', 'BF16'] as const).map((q) => (
              <button key={q} className={`toggle-option ${entry.kvCacheType === q ? 'active' : ''}`} onClick={() => onUpdate({ ...entry, kvCacheType: q })}>
                {q}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Context Length */}
      <div className="mb-3">
        <Tooltip text={`Maximum tokens per request. ${entry.model.maxPositionEmbeddings ? `This model supports up to ${(entry.model.maxPositionEmbeddings / 1024).toFixed(0)}K tokens.` : 'Larger context = more KV cache = lower batch size.'}`}>
          <label className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--color-text-secondary)' }}>
            Max Context Tokens
          </label>
        </Tooltip>
        <input
          type="range" className="glass-slider mt-1" min={0} max={contextSteps.length - 1} step={1}
          value={ctxIdx}
          onChange={(e) => onUpdate({ ...entry, maxContextTokens: contextSteps[Number(e.target.value)] })}
        />
        <div className="flex justify-between text-xs mt-0.5" style={{ color: 'var(--color-text-tertiary)' }}>
          <span>{(contextSteps[0] / 1024).toFixed(0)}K</span>
          <span className="font-medium" style={{ color: 'var(--color-accent-cyan)' }}>
            {entry.maxContextTokens >= 1048576 ? `${(entry.maxContextTokens / 1048576).toFixed(0)}M` : `${(entry.maxContextTokens / 1024).toFixed(0)}K`}
          </span>
          <span>{(contextSteps[contextSteps.length - 1] / 1024).toFixed(0)}K</span>
        </div>
      </div>

      {/* Target Concurrency */}
      <div className="mb-3">
        <Tooltip text="Total simultaneous requests across all replicas for this model. The system auto-provisions enough replicas based on the optimal batch size.">
          <label className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--color-text-secondary)' }}>
            Target Concurrency
          </label>
        </Tooltip>
        <div className="flex items-center gap-2 mt-1">
          <input type="range" className="glass-slider flex-1" min={1} max={1000} value={entry.targetConcurrency} onChange={(e) => onUpdate({ ...entry, targetConcurrency: Number(e.target.value) })} />
          <input type="number" className="glass-input" value={entry.targetConcurrency} onChange={(e) => onUpdate({ ...entry, targetConcurrency: Math.max(1, Number(e.target.value)) })} min={1} max={10000} style={{ width: '64px', flex: 'none', textAlign: 'center' }} />
        </div>
      </div>

      {/* Advanced Latency Tuning */}
      <button
        className="text-xs mb-2 transition-colors"
        style={{ color: 'var(--color-text-tertiary)' }}
        onClick={() => setShowAdvanced(!showAdvanced)}
      >
        {showAdvanced ? '▾' : '▸'} Advanced Latency Tuning
      </button>

      {showAdvanced && (
        <div className="mb-3 p-3 rounded-lg" style={{ background: 'oklch(1 0 0 / 0.02)', border: '1px solid oklch(1 0 0 / 0.04)' }}>
          <div className="flex items-center justify-between mb-2">
            <Tooltip text="Override the auto-calculated batch size. Use a lower value to reduce time-to-first-token (latency) at the cost of more replicas. Leave unchecked for optimal throughput.">
              <label className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--color-text-secondary)' }}>
                Manual Batch Override
              </label>
            </Tooltip>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={hasBatchOverride}
                onChange={(e) => {
                  if (e.target.checked) {
                    onUpdate({ ...entry, batchSizeOverride: results?.optimalBatchSize ?? 1 });
                  } else {
                    const { batchSizeOverride: _, ...rest } = entry;
                    onUpdate(rest as ModelEntry);
                  }
                }}
                className="w-3.5 h-3.5 rounded"
              />
              <span className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>Enable</span>
            </label>
          </div>
          {hasBatchOverride && (
            <div className="flex items-center gap-2">
              <input
                type="range" className="glass-slider flex-1"
                min={1} max={results?.optimalBatchSize ?? 64} step={1}
                value={entry.batchSizeOverride || 1}
                onChange={(e) => onUpdate({ ...entry, batchSizeOverride: Number(e.target.value) })}
              />
              <input
                type="number" className="glass-input"
                value={entry.batchSizeOverride || 1}
                onChange={(e) => onUpdate({ ...entry, batchSizeOverride: Math.max(1, Number(e.target.value)) })}
                min={1} max={results?.optimalBatchSize ?? 9999}
                style={{ width: '64px', flex: 'none', textAlign: 'center' }}
              />
              <span className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
                / {results?.optimalBatchSize ?? '—'}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Auto-Calculated Results */}
      {results && entry.gpuId && (
        <div
          className="mt-3 pt-3"
          style={{ borderTop: '1px solid oklch(1 0 0 / 0.06)' }}
        >
          <div className="grid grid-cols-4 gap-2 text-center">
            <div>
              <p className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>Optimal Batch</p>
              <p className="text-sm font-bold" style={{ color: 'var(--color-accent-emerald)' }}>
                {results.optimalBatchSize}
                {hasBatchOverride && (
                  <span className="text-xs font-normal ml-1" style={{ color: 'var(--color-accent-amber)' }}>→{results.effectiveBatchSize}</span>
                )}
              </p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>TP×PP</p>
              <p className="text-sm font-bold" style={{ color: 'var(--color-accent-cyan)' }}>{results.tpSize}×{results.ppSize}</p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>Replicas</p>
              <p className="text-sm font-bold" style={{ color: 'var(--color-text-primary)' }}>{results.replicas}</p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>GPUs</p>
              <p className="text-sm font-bold" style={{ color: 'var(--color-text-primary)' }}>{results.totalGpus}</p>
            </div>
          </div>
          {/* VRAM breakdown detail row */}
          <div className="mt-2 flex items-center justify-between text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
            <span>Weights: {formatGiB(results.totalWeightsGiB)}</span>
            <span>KV free: {formatGiB(results.vramLeftForKvGiB)}</span>
            <span>KV/user: {formatGiB(results.kvCachePerUserGiB)}</span>
          </div>
        </div>
      )}
    </div>
  );
}
