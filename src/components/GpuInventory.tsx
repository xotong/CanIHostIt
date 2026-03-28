'use client';

import React, { useState } from 'react';
import type { GpuSpec } from '@/lib/types';
import { GPU_TEMPLATES, createFromTemplate, createGpuSpec } from '@/lib/gpu-database';

interface GpuInventoryProps {
  gpus: GpuSpec[];
  onAdd: (gpu: GpuSpec) => void;
  onRemove: (id: string) => void;
  onUpdate: (gpu: GpuSpec) => void;
}

export default function GpuInventory({ gpus, onAdd, onRemove, onUpdate }: GpuInventoryProps) {
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState({ name: '', vramGiB: 80, utilization: 90, maxGpusPerNode: 8 });

  const handleAddCustom = () => {
    const gpu = createGpuSpec({
      name: form.name || 'Custom GPU',
      vramGiB: form.vramGiB,
      utilization: form.utilization / 100,
      maxGpusPerNode: form.maxGpusPerNode,
    });
    onAdd(gpu);
    setShowForm(false);
    setForm({ name: '', vramGiB: 80, utilization: 90, maxGpusPerNode: 8 });
  };

  const handleTemplate = (template: Omit<GpuSpec, 'id'>) => {
    onAdd(createFromTemplate(template));
  };

  const handleEdit = (gpu: GpuSpec) => {
    setEditingId(gpu.id);
    setForm({
      name: gpu.name,
      vramGiB: gpu.vramGiB,
      utilization: Math.round(gpu.utilization * 100),
      maxGpusPerNode: gpu.maxGpusPerNode,
    });
  };

  const handleSaveEdit = (id: string) => {
    const updated = createGpuSpec({
      name: form.name,
      vramGiB: form.vramGiB,
      utilization: form.utilization / 100,
      maxGpusPerNode: form.maxGpusPerNode,
    });
    updated.id = id;
    onUpdate(updated);
    setEditingId(null);
  };

  return (
    <div className="flex flex-col gap-3">
      {/* GPU List */}
      {gpus.map((gpu) => (
        <div
          key={gpu.id}
          className="metric-card flex items-center justify-between gap-3"
          style={{ padding: '12px 16px' }}
        >
          {editingId === gpu.id ? (
            <div className="flex-1 grid grid-cols-2 gap-2">
              <input className="glass-input col-span-2" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="GPU Name" />
              <input className="glass-input" type="number" value={form.vramGiB} onChange={(e) => setForm({ ...form, vramGiB: Number(e.target.value) })} min={1} />
              <input className="glass-input" type="number" value={form.utilization} onChange={(e) => setForm({ ...form, utilization: Number(e.target.value) })} min={10} max={100} />
              <div className="col-span-2 flex gap-2">
                <button className="toggle-option active flex-1" onClick={() => handleSaveEdit(gpu.id)}>Save</button>
                <button className="toggle-option flex-1" onClick={() => setEditingId(null)}>Cancel</button>
              </div>
            </div>
          ) : (
            <>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate" style={{ color: 'var(--color-text-primary)' }}>
                  {gpu.name}
                </p>
                <p className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
                  {gpu.vramGiB} GiB · {Math.round(gpu.utilization * 100)}% util · {gpu.maxGpusPerNode} GPUs/node
                </p>
              </div>
              <div className="flex gap-1">
                <button
                  className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors"
                  style={{ background: 'oklch(1 0 0 / 0.04)', color: 'var(--color-text-tertiary)' }}
                  onClick={() => handleEdit(gpu)}
                  title="Edit"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                </button>
                <button
                  className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors"
                  style={{ background: 'oklch(0.65 0.22 25 / 0.1)', color: 'var(--color-danger)' }}
                  onClick={() => onRemove(gpu.id)}
                  title="Remove"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6 6 18M6 6l12 12"/></svg>
                </button>
              </div>
            </>
          )}
        </div>
      ))}

      {gpus.length === 0 && !showForm && (
        <p className="text-xs text-center py-3" style={{ color: 'var(--color-text-tertiary)' }}>
          No GPUs added yet. Add from templates or create custom.
        </p>
      )}

      {/* Quick-add Templates */}
      {!showForm && (
        <div className="flex flex-wrap gap-1.5">
          {GPU_TEMPLATES.map((t) => {
            const alreadyAdded = gpus.some((g) => g.name === t.name);
            return (
              <button
                key={t.name}
                className="text-xs px-2.5 py-1.5 rounded-lg transition-all"
                style={{
                  background: alreadyAdded ? 'oklch(1 0 0 / 0.02)' : 'oklch(1 0 0 / 0.04)',
                  color: alreadyAdded ? 'var(--color-text-tertiary)' : 'var(--color-text-secondary)',
                  border: '1px solid oklch(1 0 0 / 0.06)',
                  opacity: alreadyAdded ? 0.5 : 1,
                }}
                onClick={() => !alreadyAdded && handleTemplate(t)}
                disabled={alreadyAdded}
              >
                + {t.name.replace('NVIDIA ', '').replace('AMD ', '')}
              </button>
            );
          })}
        </div>
      )}

      {/* Custom GPU Form */}
      {showForm ? (
        <div className="metric-card flex flex-col gap-2" style={{ padding: '14px 16px' }}>
          <p className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--color-text-secondary)' }}>
            Custom GPU
          </p>
          <input className="glass-input" placeholder="GPU Name (e.g. NVIDIA L40S)" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          <div className="grid grid-cols-3 gap-2">
            <div>
              <label className="text-xs mb-0.5 block" style={{ color: 'var(--color-text-tertiary)' }}>VRAM (GiB)</label>
              <input className="glass-input" type="number" value={form.vramGiB} onChange={(e) => setForm({ ...form, vramGiB: Number(e.target.value) })} min={1} />
            </div>
            <div>
              <label className="text-xs mb-0.5 block" style={{ color: 'var(--color-text-tertiary)' }}>Util %</label>
              <input className="glass-input" type="number" value={form.utilization} onChange={(e) => setForm({ ...form, utilization: Number(e.target.value) })} min={10} max={100} />
            </div>
            <div>
              <label className="text-xs mb-0.5 block" style={{ color: 'var(--color-text-tertiary)' }}>GPUs/Node</label>
              <input className="glass-input" type="number" value={form.maxGpusPerNode} onChange={(e) => setForm({ ...form, maxGpusPerNode: Number(e.target.value) })} min={1} max={16} />
            </div>
          </div>
          <div className="flex gap-2">
            <button className="toggle-option active flex-1" onClick={handleAddCustom}>Add GPU</button>
            <button className="toggle-option flex-1" onClick={() => setShowForm(false)}>Cancel</button>
          </div>
        </div>
      ) : (
        <button
          className="text-xs px-3 py-2 rounded-lg transition-all"
          style={{
            background: 'oklch(1 0 0 / 0.04)',
            color: 'var(--color-text-secondary)',
            border: '1px dashed oklch(1 0 0 / 0.1)',
          }}
          onClick={() => setShowForm(true)}
        >
          + Add Custom GPU
        </button>
      )}
    </div>
  );
}
