// ============================================================================
// HuggingFace API Integration
// ============================================================================
// Fetches model info from HuggingFace Hub API:
//   1. Search models: GET /api/models?search=...
//   2. Model info:    GET /api/models/{id} → safetensors.parameters
//   3. Full config:   GET /{id}/resolve/main/config.json → architecture details
// ============================================================================

import type {
  ModelSpec,
  HfSearchResult,
  HfModelInfo,
  HfConfigJson,
} from './types';

const HF_API_BASE = 'https://huggingface.co/api';
const HF_BASE = 'https://huggingface.co';
const TIMEOUT_MS = 10000;

// ─── Search ────────────────────────────────────────────────────────────────────

export async function searchHuggingFaceModels(
  query: string,
  limit: number = 8
): Promise<HfSearchResult[]> {
  if (!query || query.length < 2) return [];

  try {
    const params = new URLSearchParams({
      search: query,
      sort: 'downloads',
      direction: '-1',
      limit: limit.toString(),
    });

    const response = await fetch(`${HF_API_BASE}/models?${params}`, {
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });

    if (!response.ok) return [];

    const data: HfSearchResult[] = await response.json();
    return data.map((m) => ({
      id: m.id,
      modelId: m.modelId || m.id,
      downloads: m.downloads || 0,
      likes: m.likes || 0,
      pipeline_tag: m.pipeline_tag,
      library_name: m.library_name,
      tags: m.tags,
    }));
  } catch {
    return [];
  }
}

// ─── Model Info (safetensors params) ───────────────────────────────────────────

export async function fetchModelInfo(modelId: string): Promise<HfModelInfo | null> {
  try {
    const response = await fetch(`${HF_API_BASE}/models/${modelId}`, {
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });

    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

// ─── Config JSON (architecture details) ────────────────────────────────────────

export async function fetchConfigJson(modelId: string): Promise<HfConfigJson | null> {
  try {
    const response = await fetch(
      `${HF_BASE}/${modelId}/resolve/main/config.json`,
      { signal: AbortSignal.timeout(TIMEOUT_MS) }
    );

    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

// ─── Combined: Fetch everything and build a ModelSpec ───────────────────────────

export interface FetchModelResult {
  spec: ModelSpec;
  warnings: string[];
}

export async function fetchFullModelSpec(modelId: string): Promise<FetchModelResult | null> {
  // Fetch both in parallel
  const [modelInfo, configJson] = await Promise.all([
    fetchModelInfo(modelId),
    fetchConfigJson(modelId),
  ]);

  const warnings: string[] = [];

  if (!modelInfo && !configJson) {
    return null;
  }

  // ── Extract total params from safetensors metadata ──
  let totalParams_B = 0;
  if (modelInfo?.safetensors?.total) {
    totalParams_B = modelInfo.safetensors.total / 1_000_000_000;
  } else if (modelInfo?.safetensors?.parameters) {
    const paramValues = Object.values(modelInfo.safetensors.parameters);
    if (paramValues.length > 0) {
      totalParams_B = Math.max(...paramValues) / 1_000_000_000;
    }
  }

  if (totalParams_B === 0) {
    warnings.push('Could not determine total parameters. Please enter manually.');
  }

  // ── Extract architecture from config.json ──
  // VLMs (e.g. Qwen3.5-9B) nest text model config under text_config
  const tc = (configJson as Record<string, unknown>)?.text_config as HfConfigJson | undefined;
  const cfg = configJson;

  const layers =
    cfg?.num_hidden_layers ?? tc?.num_hidden_layers ??
    cfg?.n_layer ?? tc?.n_layer ??
    null;

  const kvHeads =
    cfg?.num_key_value_heads ?? tc?.num_key_value_heads ??
    cfg?.n_head_kv ?? tc?.n_head_kv ??
    cfg?.num_attention_heads ?? tc?.num_attention_heads ??
    null;

  let headDim = cfg?.head_dim ?? tc?.head_dim ?? null;
  if (!headDim) {
    const hs = (cfg?.hidden_size ?? tc?.hidden_size) as number | undefined;
    const nah = (cfg?.num_attention_heads ?? tc?.num_attention_heads) as number | undefined;
    if (hs && nah) headDim = Math.floor(hs / nah);
  }

  const maxPositionEmbeddings =
    cfg?.max_position_embeddings ?? tc?.max_position_embeddings ??
    cfg?.max_seq_len ?? tc?.max_seq_len ??
    cfg?.model_max_length ?? tc?.model_max_length ??
    null;

  if (layers === null) warnings.push('Could not determine layer count.');
  if (kvHeads === null) warnings.push('Could not determine KV head count.');
  if (headDim === null) warnings.push('Could not determine head dimension.');

  const displayName = modelId.split('/').pop() || modelId;

  const spec: ModelSpec = {
    name: displayName,
    huggingfaceId: modelId,
    totalParams_B: Math.round(totalParams_B * 100) / 100,
    layers: layers ?? 0,
    kvHeads: kvHeads ?? 0,
    headDim: headDim ?? 0,
    maxPositionEmbeddings: maxPositionEmbeddings ?? undefined,
    description: modelInfo?.config?.architectures?.[0]
      ? `${modelInfo.config.architectures[0]} · ${modelId}`
      : modelId,
  };

  return { spec, warnings };
}

// ─── Format download count ─────────────────────────────────────────────────────

export function formatDownloads(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toString();
}
