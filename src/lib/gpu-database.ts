// ============================================================================
// GPU Inventory Manager — localStorage-persisted, no hardcoded presets
// ============================================================================

import type { GpuSpec } from './types';

const STORAGE_KEY = 'canihostit-gpu-inventory';

/** Generate powers of 2 up to max */
function generateValidTpSizes(maxGpus: number): number[] {
  const sizes: number[] = [];
  for (let i = 1; i <= maxGpus; i *= 2) sizes.push(i);
  return sizes;
}

/** Quick-add GPU templates (convenience, not presets) */
export const GPU_TEMPLATES: Omit<GpuSpec, 'id'>[] = [
  { name: 'NVIDIA A100 80GB SXM', vramGiB: 80, utilization: 0.90, maxGpusPerNode: 8, validTpSizes: [1, 2, 4, 8] },
  { name: 'NVIDIA H100 80GB SXM', vramGiB: 80, utilization: 0.90, maxGpusPerNode: 8, validTpSizes: [1, 2, 4, 8] },
  { name: 'NVIDIA H200 141GB SXM', vramGiB: 141, utilization: 0.90, maxGpusPerNode: 8, validTpSizes: [1, 2, 4, 8] },
  { name: 'NVIDIA B200 192GB SXM', vramGiB: 192, utilization: 0.90, maxGpusPerNode: 8, validTpSizes: [1, 2, 4, 8] },
  { name: 'AMD MI300X 192GB', vramGiB: 192, utilization: 0.90, maxGpusPerNode: 8, validTpSizes: [1, 2, 4, 8] },
  { name: 'AMD MI325X 256GB', vramGiB: 256, utilization: 0.90, maxGpusPerNode: 8, validTpSizes: [1, 2, 4, 8] },
];

/** Generate a unique ID */
export function generateId(): string {
  return `gpu_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

/** Load GPU inventory from localStorage */
export function loadGpuInventory(): GpuSpec[] {
  if (typeof window === 'undefined') return [];
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return [];
    return JSON.parse(stored);
  } catch {
    return [];
  }
}

/** Save GPU inventory to localStorage */
export function saveGpuInventory(gpus: GpuSpec[]): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(gpus));
  } catch {
    // localStorage full or unavailable
  }
}

/** Create a new GPU spec from form data */
export function createGpuSpec(data: {
  name: string;
  vramGiB: number;
  utilization: number;
  maxGpusPerNode: number;
}): GpuSpec {
  return {
    id: generateId(),
    name: data.name,
    vramGiB: data.vramGiB,
    utilization: data.utilization,
    maxGpusPerNode: data.maxGpusPerNode,
    validTpSizes: generateValidTpSizes(data.maxGpusPerNode),
  };
}

/** Create GPU from template */
export function createFromTemplate(template: Omit<GpuSpec, 'id'>): GpuSpec {
  return { ...template, id: generateId() };
}
