// ============================================================================
// Shared Types for AI Infrastructure Capacity Planner
// ============================================================================

/** GPU hardware specification */
export interface GpuSpec {
  id: string;
  name: string;
  vramGiB: number;
  utilization: number;       // 0-1 (vLLM --gpu-memory-utilization)
  maxGpusPerNode: number;
  validTpSizes: number[];    // e.g. [1, 2, 4, 8]
}

/** Model architecture specification */
export interface ModelSpec {
  name: string;
  huggingfaceId?: string;
  totalParams_B: number;     // Total parameters in billions
  layers: number;
  kvHeads: number;
  headDim: number;
  maxPositionEmbeddings?: number; // from HF config
  description?: string;
}

/** A single model deployment entry in the fleet */
export interface ModelEntry {
  id: string;
  model: ModelSpec;
  gpuId: string;             // references GpuSpec.id
  quantization: 'FP8' | 'BF16';
  kvCacheType: 'FP8' | 'BF16';  // KV cache precision
  maxContextTokens: number;
  batchSizeOverride?: number;    // manual override, auto-derived if absent
  targetConcurrency: number;
}

/** Calculation results for a single model entry */
export interface ModelResults {
  entryId: string;
  modelName: string;
  gpuName: string;

  // Weight breakdown
  baseWeightsGiB: number;
  frameworkOverheadGiB: number;
  totalWeightsGiB: number;

  // KV Cache (auto-optimizer)
  kvCachePerUserGiB: number;
  vramLeftForKvGiB: number;
  optimalBatchSize: number;
  effectiveBatchSize: number; // = override || optimal
  kvCachePerReplicaGiB: number;

  // Per-replica
  totalVramPerReplicaGiB: number;

  // GPU splitting
  usableVramPerGpuGiB: number;
  minGpusRequired: number;
  tpSize: number;
  ppSize: number;
  gpusPerReplica: number;

  // Fleet sizing for this model
  replicas: number;
  totalGpus: number;
  totalNodes: number;
  totalVramGiB: number;
}

/** Aggregated fleet totals across all models */
export interface FleetTotals {
  totalGpus: number;
  totalNodes: number;
  totalVramGiB: number;
  totalRacks: number;
  totalPowerKw: number;
  modelResults: ModelResults[];
}

/** Tooltip definition */
export interface TooltipInfo {
  field: string;
  text: string;
}

/** HuggingFace search result */
export interface HfSearchResult {
  id: string;
  modelId: string;
  downloads: number;
  likes: number;
  pipeline_tag?: string;
  library_name?: string;
  tags?: string[];
}

/** HuggingFace model API response (simplified) */
export interface HfModelInfo {
  id: string;
  modelId: string;
  safetensors?: {
    parameters?: Record<string, number>;
    total?: number;
  };
  config?: {
    architectures?: string[];
    model_type?: string;
  };
}

/** HuggingFace config.json (partial) */
export interface HfConfigJson {
  num_hidden_layers?: number;
  num_key_value_heads?: number;
  head_dim?: number;
  hidden_size?: number;
  num_attention_heads?: number;
  max_position_embeddings?: number;
  max_seq_len?: number;
  model_max_length?: number;
  n_layer?: number;
  n_head_kv?: number;
  [key: string]: unknown;
}
