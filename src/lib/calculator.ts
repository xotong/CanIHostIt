// ============================================================================
// AI Infrastructure Capacity Planner — Mathematical Engine (V2.1)
// ============================================================================
// Auto-Optimizer: batch size is derived from leftover VRAM after weights.
// TP/PP is determined by weights alone, then batch fills remaining space.
// ============================================================================

import type { GpuSpec, ModelEntry, ModelResults, FleetTotals } from './types';

// ============================================================================
// Constants
// ============================================================================

const BYTES_PER_GIB = 1024 ** 3;
const VLLM_FRAMEWORK_OVERHEAD = 0.20;

export function getBytesPerParam(quantization: 'FP8' | 'BF16'): number {
  return quantization === 'FP8' ? 1 : 2;
}

export function getBytesPerKvElement(kvCacheType: 'FP8' | 'BF16'): number {
  return kvCacheType === 'FP8' ? 1 : 2;
}

// ============================================================================
// Core Calculations
// ============================================================================

/** Weight VRAM in GiB */
export function calculateBaseWeightsGiB(totalParams_B: number, bytesPerParam: number): number {
  return (totalParams_B * 1_000_000_000 * bytesPerParam) / BYTES_PER_GIB;
}

/** Weights + 20% vLLM framework overhead */
export function calculateTotalWeightsGiB(baseWeightsGiB: number): number {
  return baseWeightsGiB * (1 + VLLM_FRAMEWORK_OVERHEAD);
}

/** KV cache for a SINGLE user at given context length */
export function calculateKvCachePerUserGiB(
  kvHeads: number, headDim: number, layers: number,
  maxContext: number, bytesPerKvElement: number
): number {
  // 2 = key + value tensors
  return (2 * kvHeads * headDim * bytesPerKvElement * layers * maxContext) / BYTES_PER_GIB;
}

/** Usable VRAM per GPU after utilization cap */
export function calculateUsableVramPerGpu(gpu: GpuSpec): number {
  return gpu.vramGiB * gpu.utilization;
}

/** Find the smallest valid TP size >= minGpus */
function findNearestTpSize(minGpus: number, validTpSizes: number[]): number {
  const sorted = [...validTpSizes].sort((a, b) => a - b);
  for (const tp of sorted) {
    if (tp >= minGpus) return tp;
  }
  return sorted[sorted.length - 1];
}

/** Determine TP/PP split based on WEIGHT VRAM only (not KV cache) */
export function calculateTpPp(
  totalWeightsGiB: number, gpu: GpuSpec
): { tpSize: number; ppSize: number; gpusPerReplica: number; minGpusRequired: number } {
  const usableVram = calculateUsableVramPerGpu(gpu);
  const minGpus = Math.max(1, Math.ceil(totalWeightsGiB / usableVram));

  let tpSize: number;
  let ppSize: number;

  if (minGpus <= gpu.maxGpusPerNode) {
    tpSize = findNearestTpSize(minGpus, gpu.validTpSizes);
    ppSize = 1;
  } else {
    tpSize = gpu.maxGpusPerNode;
    ppSize = Math.ceil(minGpus / gpu.maxGpusPerNode);
  }

  return { tpSize, ppSize, gpusPerReplica: tpSize * ppSize, minGpusRequired: minGpus };
}

// ============================================================================
// Per-Model Auto-Optimized Calculation
// ============================================================================

export function calculateForModel(
  entry: ModelEntry,
  gpu: GpuSpec,
  totalDevelopers: number,
  peakActiveRate: number
): ModelResults {
  const { model, quantization, kvCacheType, maxContextTokens, agenticMultiplier } = entry;

  // ─── Step 1: Calculate weights ───────────────────────────
  const bytesPerParam = getBytesPerParam(quantization);
  const baseWeightsGiB = calculateBaseWeightsGiB(model.totalParams_B, bytesPerParam);
  const frameworkOverheadGiB = baseWeightsGiB * VLLM_FRAMEWORK_OVERHEAD;
  const totalWeightsGiB = calculateTotalWeightsGiB(baseWeightsGiB);

  // ─── Step 2: Determine TP/PP from weights only ──────────
  const usableVramPerGpuGiB = calculateUsableVramPerGpu(gpu);
  const { tpSize, ppSize, gpusPerReplica, minGpusRequired } = calculateTpPp(totalWeightsGiB, gpu);

  // ─── Step 3: Auto-derive optimal batch size ─────────────
  const totalUsableVramPerReplica = gpusPerReplica * usableVramPerGpuGiB;
  const vramLeftForKvGiB = Math.max(0, totalUsableVramPerReplica - totalWeightsGiB);

  const bytesPerKv = getBytesPerKvElement(kvCacheType);
  const kvCachePerUserGiB = calculateKvCachePerUserGiB(
    model.kvHeads, model.headDim, model.layers, maxContextTokens, bytesPerKv
  );

  const optimalBatchSize = kvCachePerUserGiB > 0
    ? Math.max(1, Math.floor(vramLeftForKvGiB / kvCachePerUserGiB))
    : 1;

  // Allow manual override (for latency tuning)
  const effectiveBatchSize = entry.batchSizeOverride ?? optimalBatchSize;

  // ─── Step 4: Calculate replicas from modeled concurrency ─
  const kvCachePerReplicaGiB = kvCachePerUserGiB * effectiveBatchSize;
  const totalVramPerReplicaGiB = totalWeightsGiB + kvCachePerReplicaGiB;
  const peakActiveUsers = Math.max(1, Math.ceil(totalDevelopers * peakActiveRate));
  const modeledConcurrency = Math.max(1, Math.ceil(peakActiveUsers * agenticMultiplier));
  const replicas = Math.max(1, Math.ceil(modeledConcurrency / effectiveBatchSize));
  const totalGpus = replicas * gpusPerReplica;
  const totalNodes = Math.ceil(totalGpus / gpu.maxGpusPerNode);

  return {
    entryId: entry.id,
    modelName: model.name,
    gpuName: gpu.name,
    agenticMultiplier,
    peakActiveUsers,
    baseWeightsGiB, frameworkOverheadGiB, totalWeightsGiB,
    kvCachePerUserGiB, vramLeftForKvGiB,
    optimalBatchSize, effectiveBatchSize,
    kvCachePerReplicaGiB, totalVramPerReplicaGiB,
    usableVramPerGpuGiB, minGpusRequired,
    tpSize, ppSize, gpusPerReplica,
    modeledConcurrency,
    replicas, totalGpus, totalNodes,
    totalVramGiB: totalGpus * gpu.vramGiB,
  };
}

// ============================================================================
// Fleet Aggregation
// ============================================================================

export function calculateFleetTotals(
  entries: ModelEntry[],
  gpuInventory: GpuSpec[],
  totalDevelopers: number,
  peakActiveRate: number,
  rackPowerBudgetKw: number,
  nodePowerKw: number
): FleetTotals {
  const gpuMap = new Map(gpuInventory.map((g) => [g.id, g]));

  const modelResults: ModelResults[] = entries
    .map((entry) => {
      const gpu = gpuMap.get(entry.gpuId);
      if (!gpu) return null;
      return calculateForModel(entry, gpu, totalDevelopers, peakActiveRate);
    })
    .filter((r): r is ModelResults => r !== null);

  const totalGpus = modelResults.reduce((sum, r) => sum + r.totalGpus, 0);
  const totalNodes = modelResults.reduce((sum, r) => sum + r.totalNodes, 0);
  const totalVramGiB = modelResults.reduce((sum, r) => sum + r.totalVramGiB, 0);

  const nodesPerRack = nodePowerKw > 0 ? Math.max(1, Math.floor(rackPowerBudgetKw / nodePowerKw)) : 1;
  const totalRacks = Math.ceil(totalNodes / nodesPerRack);
  const totalPowerKw = totalNodes * nodePowerKw;

  return { totalGpus, totalNodes, totalVramGiB, totalRacks, totalPowerKw, modelResults };
}
