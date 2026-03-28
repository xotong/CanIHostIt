module.exports = [
"[project]/Documents/GitHub/CanIHostIt/src/lib/calculator.ts [app-ssr] (ecmascript)", ((__turbopack_context__) => {
"use strict";

// ============================================================================
// AI Infrastructure Capacity Planner — Mathematical Engine (V2.1)
// ============================================================================
// Auto-Optimizer: batch size is derived from leftover VRAM after weights.
// TP/PP is determined by weights alone, then batch fills remaining space.
// ============================================================================
__turbopack_context__.s([
    "calculateBaseWeightsGiB",
    ()=>calculateBaseWeightsGiB,
    "calculateFleetTotals",
    ()=>calculateFleetTotals,
    "calculateForModel",
    ()=>calculateForModel,
    "calculateKvCachePerUserGiB",
    ()=>calculateKvCachePerUserGiB,
    "calculateTotalWeightsGiB",
    ()=>calculateTotalWeightsGiB,
    "calculateTpPp",
    ()=>calculateTpPp,
    "calculateUsableVramPerGpu",
    ()=>calculateUsableVramPerGpu,
    "getBytesPerKvElement",
    ()=>getBytesPerKvElement,
    "getBytesPerParam",
    ()=>getBytesPerParam
]);
// ============================================================================
// Constants
// ============================================================================
const BYTES_PER_GIB = 1024 ** 3;
const VLLM_FRAMEWORK_OVERHEAD = 0.20;
function getBytesPerParam(quantization) {
    return quantization === 'FP8' ? 1 : 2;
}
function getBytesPerKvElement(kvCacheType) {
    return kvCacheType === 'FP8' ? 1 : 2;
}
function calculateBaseWeightsGiB(totalParams_B, bytesPerParam) {
    return totalParams_B * 1_000_000_000 * bytesPerParam / BYTES_PER_GIB;
}
function calculateTotalWeightsGiB(baseWeightsGiB) {
    return baseWeightsGiB * (1 + VLLM_FRAMEWORK_OVERHEAD);
}
function calculateKvCachePerUserGiB(kvHeads, headDim, layers, maxContext, bytesPerKvElement) {
    // 2 = key + value tensors
    return 2 * kvHeads * headDim * bytesPerKvElement * layers * maxContext / BYTES_PER_GIB;
}
function calculateUsableVramPerGpu(gpu) {
    return gpu.vramGiB * gpu.utilization;
}
/** Find the smallest valid TP size >= minGpus */ function findNearestTpSize(minGpus, validTpSizes) {
    const sorted = [
        ...validTpSizes
    ].sort((a, b)=>a - b);
    for (const tp of sorted){
        if (tp >= minGpus) return tp;
    }
    return sorted[sorted.length - 1];
}
function calculateTpPp(totalWeightsGiB, gpu) {
    const usableVram = calculateUsableVramPerGpu(gpu);
    const minGpus = Math.max(1, Math.ceil(totalWeightsGiB / usableVram));
    let tpSize;
    let ppSize;
    if (minGpus <= gpu.maxGpusPerNode) {
        tpSize = findNearestTpSize(minGpus, gpu.validTpSizes);
        ppSize = 1;
    } else {
        tpSize = gpu.maxGpusPerNode;
        ppSize = Math.ceil(minGpus / gpu.maxGpusPerNode);
    }
    return {
        tpSize,
        ppSize,
        gpusPerReplica: tpSize * ppSize,
        minGpusRequired: minGpus
    };
}
function calculateForModel(entry, gpu) {
    const { model, quantization, kvCacheType, maxContextTokens, targetConcurrency } = entry;
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
    const kvCachePerUserGiB = calculateKvCachePerUserGiB(model.kvHeads, model.headDim, model.layers, maxContextTokens, bytesPerKv);
    const optimalBatchSize = kvCachePerUserGiB > 0 ? Math.max(1, Math.floor(vramLeftForKvGiB / kvCachePerUserGiB)) : 1;
    // Allow manual override (for latency tuning)
    const effectiveBatchSize = entry.batchSizeOverride ?? optimalBatchSize;
    // ─── Step 4: Calculate replicas from concurrency ────────
    const kvCachePerReplicaGiB = kvCachePerUserGiB * effectiveBatchSize;
    const totalVramPerReplicaGiB = totalWeightsGiB + kvCachePerReplicaGiB;
    const replicas = Math.max(1, Math.ceil(targetConcurrency / effectiveBatchSize));
    const totalGpus = replicas * gpusPerReplica;
    const totalNodes = Math.ceil(totalGpus / gpu.maxGpusPerNode);
    return {
        entryId: entry.id,
        modelName: model.name,
        gpuName: gpu.name,
        baseWeightsGiB,
        frameworkOverheadGiB,
        totalWeightsGiB,
        kvCachePerUserGiB,
        vramLeftForKvGiB,
        optimalBatchSize,
        effectiveBatchSize,
        kvCachePerReplicaGiB,
        totalVramPerReplicaGiB,
        usableVramPerGpuGiB,
        minGpusRequired,
        tpSize,
        ppSize,
        gpusPerReplica,
        replicas,
        totalGpus,
        totalNodes,
        totalVramGiB: totalGpus * gpu.vramGiB
    };
}
function calculateFleetTotals(entries, gpuInventory, rackPowerBudgetKw, nodePowerKw) {
    const gpuMap = new Map(gpuInventory.map((g)=>[
            g.id,
            g
        ]));
    const modelResults = entries.map((entry)=>{
        const gpu = gpuMap.get(entry.gpuId);
        if (!gpu) return null;
        return calculateForModel(entry, gpu);
    }).filter((r)=>r !== null);
    const totalGpus = modelResults.reduce((sum, r)=>sum + r.totalGpus, 0);
    const totalNodes = modelResults.reduce((sum, r)=>sum + r.totalNodes, 0);
    const totalVramGiB = modelResults.reduce((sum, r)=>sum + r.totalVramGiB, 0);
    const nodesPerRack = nodePowerKw > 0 ? Math.max(1, Math.floor(rackPowerBudgetKw / nodePowerKw)) : 1;
    const totalRacks = Math.ceil(totalNodes / nodesPerRack);
    const totalPowerKw = totalNodes * nodePowerKw;
    return {
        totalGpus,
        totalNodes,
        totalVramGiB,
        totalRacks,
        totalPowerKw,
        modelResults
    };
}
}),
"[project]/Documents/GitHub/CanIHostIt/src/lib/gpu-database.ts [app-ssr] (ecmascript)", ((__turbopack_context__) => {
"use strict";

// ============================================================================
// GPU Inventory Manager — localStorage-persisted, no hardcoded presets
// ============================================================================
__turbopack_context__.s([
    "GPU_TEMPLATES",
    ()=>GPU_TEMPLATES,
    "createFromTemplate",
    ()=>createFromTemplate,
    "createGpuSpec",
    ()=>createGpuSpec,
    "generateId",
    ()=>generateId,
    "loadGpuInventory",
    ()=>loadGpuInventory,
    "saveGpuInventory",
    ()=>saveGpuInventory
]);
const STORAGE_KEY = 'canihostit-gpu-inventory';
/** Generate powers of 2 up to max */ function generateValidTpSizes(maxGpus) {
    const sizes = [];
    for(let i = 1; i <= maxGpus; i *= 2)sizes.push(i);
    return sizes;
}
const GPU_TEMPLATES = [
    {
        name: 'NVIDIA A100 80GB SXM',
        vramGiB: 80,
        utilization: 0.90,
        maxGpusPerNode: 8,
        validTpSizes: [
            1,
            2,
            4,
            8
        ]
    },
    {
        name: 'NVIDIA H100 80GB SXM',
        vramGiB: 80,
        utilization: 0.90,
        maxGpusPerNode: 8,
        validTpSizes: [
            1,
            2,
            4,
            8
        ]
    },
    {
        name: 'NVIDIA H200 141GB SXM',
        vramGiB: 141,
        utilization: 0.90,
        maxGpusPerNode: 8,
        validTpSizes: [
            1,
            2,
            4,
            8
        ]
    },
    {
        name: 'NVIDIA B200 192GB SXM',
        vramGiB: 192,
        utilization: 0.90,
        maxGpusPerNode: 8,
        validTpSizes: [
            1,
            2,
            4,
            8
        ]
    },
    {
        name: 'AMD MI300X 192GB',
        vramGiB: 192,
        utilization: 0.90,
        maxGpusPerNode: 8,
        validTpSizes: [
            1,
            2,
            4,
            8
        ]
    },
    {
        name: 'AMD MI325X 256GB',
        vramGiB: 256,
        utilization: 0.90,
        maxGpusPerNode: 8,
        validTpSizes: [
            1,
            2,
            4,
            8
        ]
    }
];
function generateId() {
    return `gpu_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}
function loadGpuInventory() {
    if ("TURBOPACK compile-time truthy", 1) return [];
    //TURBOPACK unreachable
    ;
}
function saveGpuInventory(gpus) {
    if ("TURBOPACK compile-time truthy", 1) return;
    //TURBOPACK unreachable
    ;
}
function createGpuSpec(data) {
    return {
        id: generateId(),
        name: data.name,
        vramGiB: data.vramGiB,
        utilization: data.utilization,
        maxGpusPerNode: data.maxGpusPerNode,
        validTpSizes: generateValidTpSizes(data.maxGpusPerNode)
    };
}
function createFromTemplate(template) {
    return {
        ...template,
        id: generateId()
    };
}
}),
"[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx [app-ssr] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>GpuInventory
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/Documents/GitHub/CanIHostIt/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/Documents/GitHub/CanIHostIt/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$lib$2f$gpu$2d$database$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/Documents/GitHub/CanIHostIt/src/lib/gpu-database.ts [app-ssr] (ecmascript)");
'use client';
;
;
;
function GpuInventory({ gpus, onAdd, onRemove, onUpdate }) {
    const [showForm, setShowForm] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(false);
    const [editingId, setEditingId] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(null);
    const [form, setForm] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])({
        name: '',
        vramGiB: 80,
        utilization: 90,
        maxGpusPerNode: 8
    });
    const handleAddCustom = ()=>{
        const gpu = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$lib$2f$gpu$2d$database$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["createGpuSpec"])({
            name: form.name || 'Custom GPU',
            vramGiB: form.vramGiB,
            utilization: form.utilization / 100,
            maxGpusPerNode: form.maxGpusPerNode
        });
        onAdd(gpu);
        setShowForm(false);
        setForm({
            name: '',
            vramGiB: 80,
            utilization: 90,
            maxGpusPerNode: 8
        });
    };
    const handleTemplate = (template)=>{
        onAdd((0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$lib$2f$gpu$2d$database$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["createFromTemplate"])(template));
    };
    const handleEdit = (gpu)=>{
        setEditingId(gpu.id);
        setForm({
            name: gpu.name,
            vramGiB: gpu.vramGiB,
            utilization: Math.round(gpu.utilization * 100),
            maxGpusPerNode: gpu.maxGpusPerNode
        });
    };
    const handleSaveEdit = (id)=>{
        const updated = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$lib$2f$gpu$2d$database$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["createGpuSpec"])({
            name: form.name,
            vramGiB: form.vramGiB,
            utilization: form.utilization / 100,
            maxGpusPerNode: form.maxGpusPerNode
        });
        updated.id = id;
        onUpdate(updated);
        setEditingId(null);
    };
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "flex flex-col gap-3",
        children: [
            gpus.map((gpu)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    className: "metric-card flex items-center justify-between gap-3",
                    style: {
                        padding: '12px 16px'
                    },
                    children: editingId === gpu.id ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex-1 grid grid-cols-2 gap-2",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                className: "glass-input col-span-2",
                                value: form.name,
                                onChange: (e)=>setForm({
                                        ...form,
                                        name: e.target.value
                                    }),
                                placeholder: "GPU Name"
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                lineNumber: 68,
                                columnNumber: 15
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                className: "glass-input",
                                type: "number",
                                value: form.vramGiB,
                                onChange: (e)=>setForm({
                                        ...form,
                                        vramGiB: Number(e.target.value)
                                    }),
                                min: 1
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                lineNumber: 69,
                                columnNumber: 15
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                className: "glass-input",
                                type: "number",
                                value: form.utilization,
                                onChange: (e)=>setForm({
                                        ...form,
                                        utilization: Number(e.target.value)
                                    }),
                                min: 10,
                                max: 100
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                lineNumber: 70,
                                columnNumber: 15
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "col-span-2 flex gap-2",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                        className: "toggle-option active flex-1",
                                        onClick: ()=>handleSaveEdit(gpu.id),
                                        children: "Save"
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                        lineNumber: 72,
                                        columnNumber: 17
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                        className: "toggle-option flex-1",
                                        onClick: ()=>setEditingId(null),
                                        children: "Cancel"
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                        lineNumber: 73,
                                        columnNumber: 17
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                lineNumber: 71,
                                columnNumber: 15
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                        lineNumber: 67,
                        columnNumber: 13
                    }, this) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Fragment"], {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex-1 min-w-0",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                        className: "text-sm font-medium truncate",
                                        style: {
                                            color: 'var(--color-text-primary)'
                                        },
                                        children: gpu.name
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                        lineNumber: 79,
                                        columnNumber: 17
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                        className: "text-xs",
                                        style: {
                                            color: 'var(--color-text-tertiary)'
                                        },
                                        children: [
                                            gpu.vramGiB,
                                            " GiB · ",
                                            Math.round(gpu.utilization * 100),
                                            "% util · ",
                                            gpu.maxGpusPerNode,
                                            " GPUs/node"
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                        lineNumber: 82,
                                        columnNumber: 17
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                lineNumber: 78,
                                columnNumber: 15
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex gap-1",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                        className: "w-7 h-7 rounded-lg flex items-center justify-center transition-colors",
                                        style: {
                                            background: 'oklch(1 0 0 / 0.04)',
                                            color: 'var(--color-text-tertiary)'
                                        },
                                        onClick: ()=>handleEdit(gpu),
                                        title: "Edit",
                                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("svg", {
                                            width: "12",
                                            height: "12",
                                            viewBox: "0 0 24 24",
                                            fill: "none",
                                            stroke: "currentColor",
                                            strokeWidth: "2",
                                            children: [
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                                                    d: "M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"
                                                }, void 0, false, {
                                                    fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                                    lineNumber: 93,
                                                    columnNumber: 117
                                                }, this),
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                                                    d: "M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"
                                                }, void 0, false, {
                                                    fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                                    lineNumber: 93,
                                                    columnNumber: 187
                                                }, this)
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                            lineNumber: 93,
                                            columnNumber: 19
                                        }, this)
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                        lineNumber: 87,
                                        columnNumber: 17
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                        className: "w-7 h-7 rounded-lg flex items-center justify-center transition-colors",
                                        style: {
                                            background: 'oklch(0.65 0.22 25 / 0.1)',
                                            color: 'var(--color-danger)'
                                        },
                                        onClick: ()=>onRemove(gpu.id),
                                        title: "Remove",
                                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("svg", {
                                            width: "12",
                                            height: "12",
                                            viewBox: "0 0 24 24",
                                            fill: "none",
                                            stroke: "currentColor",
                                            strokeWidth: "2",
                                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                                                d: "M18 6 6 18M6 6l12 12"
                                            }, void 0, false, {
                                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                                lineNumber: 101,
                                                columnNumber: 117
                                            }, this)
                                        }, void 0, false, {
                                            fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                            lineNumber: 101,
                                            columnNumber: 19
                                        }, this)
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                        lineNumber: 95,
                                        columnNumber: 17
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                lineNumber: 86,
                                columnNumber: 15
                            }, this)
                        ]
                    }, void 0, true)
                }, gpu.id, false, {
                    fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                    lineNumber: 61,
                    columnNumber: 9
                }, this)),
            gpus.length === 0 && !showForm && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                className: "text-xs text-center py-3",
                style: {
                    color: 'var(--color-text-tertiary)'
                },
                children: "No GPUs added yet. Add from templates or create custom."
            }, void 0, false, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                lineNumber: 110,
                columnNumber: 9
            }, this),
            !showForm && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "flex flex-wrap gap-1.5",
                children: __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$lib$2f$gpu$2d$database$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["GPU_TEMPLATES"].map((t)=>{
                    const alreadyAdded = gpus.some((g)=>g.name === t.name);
                    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                        className: "text-xs px-2.5 py-1.5 rounded-lg transition-all",
                        style: {
                            background: alreadyAdded ? 'oklch(1 0 0 / 0.02)' : 'oklch(1 0 0 / 0.04)',
                            color: alreadyAdded ? 'var(--color-text-tertiary)' : 'var(--color-text-secondary)',
                            border: '1px solid oklch(1 0 0 / 0.06)',
                            opacity: alreadyAdded ? 0.5 : 1
                        },
                        onClick: ()=>!alreadyAdded && handleTemplate(t),
                        disabled: alreadyAdded,
                        children: [
                            "+ ",
                            t.name.replace('NVIDIA ', '').replace('AMD ', '')
                        ]
                    }, t.name, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                        lineNumber: 121,
                        columnNumber: 15
                    }, this);
                })
            }, void 0, false, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                lineNumber: 117,
                columnNumber: 9
            }, this),
            showForm ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "metric-card flex flex-col gap-2",
                style: {
                    padding: '14px 16px'
                },
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                        className: "text-xs font-medium uppercase tracking-wider",
                        style: {
                            color: 'var(--color-text-secondary)'
                        },
                        children: "Custom GPU"
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                        lineNumber: 143,
                        columnNumber: 11
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                        className: "glass-input",
                        placeholder: "GPU Name (e.g. NVIDIA L40S)",
                        value: form.name,
                        onChange: (e)=>setForm({
                                ...form,
                                name: e.target.value
                            })
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                        lineNumber: 146,
                        columnNumber: 11
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "grid grid-cols-3 gap-2",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                                        className: "text-xs mb-0.5 block",
                                        style: {
                                            color: 'var(--color-text-tertiary)'
                                        },
                                        children: "VRAM (GiB)"
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                        lineNumber: 149,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                        className: "glass-input",
                                        type: "number",
                                        value: form.vramGiB,
                                        onChange: (e)=>setForm({
                                                ...form,
                                                vramGiB: Number(e.target.value)
                                            }),
                                        min: 1
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                        lineNumber: 150,
                                        columnNumber: 15
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                lineNumber: 148,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                                        className: "text-xs mb-0.5 block",
                                        style: {
                                            color: 'var(--color-text-tertiary)'
                                        },
                                        children: "Util %"
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                        lineNumber: 153,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                        className: "glass-input",
                                        type: "number",
                                        value: form.utilization,
                                        onChange: (e)=>setForm({
                                                ...form,
                                                utilization: Number(e.target.value)
                                            }),
                                        min: 10,
                                        max: 100
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                        lineNumber: 154,
                                        columnNumber: 15
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                lineNumber: 152,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                                        className: "text-xs mb-0.5 block",
                                        style: {
                                            color: 'var(--color-text-tertiary)'
                                        },
                                        children: "GPUs/Node"
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                        lineNumber: 157,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                        className: "glass-input",
                                        type: "number",
                                        value: form.maxGpusPerNode,
                                        onChange: (e)=>setForm({
                                                ...form,
                                                maxGpusPerNode: Number(e.target.value)
                                            }),
                                        min: 1,
                                        max: 16
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                        lineNumber: 158,
                                        columnNumber: 15
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                lineNumber: 156,
                                columnNumber: 13
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                        lineNumber: 147,
                        columnNumber: 11
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex gap-2",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                className: "toggle-option active flex-1",
                                onClick: handleAddCustom,
                                children: "Add GPU"
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                lineNumber: 162,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                className: "toggle-option flex-1",
                                onClick: ()=>setShowForm(false),
                                children: "Cancel"
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                                lineNumber: 163,
                                columnNumber: 13
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                        lineNumber: 161,
                        columnNumber: 11
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                lineNumber: 142,
                columnNumber: 9
            }, this) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                className: "text-xs px-3 py-2 rounded-lg transition-all",
                style: {
                    background: 'oklch(1 0 0 / 0.04)',
                    color: 'var(--color-text-secondary)',
                    border: '1px dashed oklch(1 0 0 / 0.1)'
                },
                onClick: ()=>setShowForm(true),
                children: "+ Add Custom GPU"
            }, void 0, false, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
                lineNumber: 167,
                columnNumber: 9
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx",
        lineNumber: 58,
        columnNumber: 5
    }, this);
}
}),
"[project]/Documents/GitHub/CanIHostIt/src/lib/model-database.ts [app-ssr] (ecmascript)", ((__turbopack_context__) => {
"use strict";

// ============================================================================
// HuggingFace API Integration
// ============================================================================
// Fetches model info from HuggingFace Hub API:
//   1. Search models: GET /api/models?search=...
//   2. Model info:    GET /api/models/{id} → safetensors.parameters
//   3. Full config:   GET /{id}/resolve/main/config.json → architecture details
// ============================================================================
__turbopack_context__.s([
    "fetchConfigJson",
    ()=>fetchConfigJson,
    "fetchFullModelSpec",
    ()=>fetchFullModelSpec,
    "fetchModelInfo",
    ()=>fetchModelInfo,
    "formatDownloads",
    ()=>formatDownloads,
    "searchHuggingFaceModels",
    ()=>searchHuggingFaceModels
]);
const HF_API_BASE = 'https://huggingface.co/api';
const HF_BASE = 'https://huggingface.co';
const TIMEOUT_MS = 10000;
async function searchHuggingFaceModels(query, limit = 8) {
    if (!query || query.length < 2) return [];
    try {
        const params = new URLSearchParams({
            search: query,
            sort: 'downloads',
            direction: '-1',
            limit: limit.toString()
        });
        const response = await fetch(`${HF_API_BASE}/models?${params}`, {
            signal: AbortSignal.timeout(TIMEOUT_MS)
        });
        if (!response.ok) return [];
        const data = await response.json();
        return data.map((m)=>({
                id: m.id,
                modelId: m.modelId || m.id,
                downloads: m.downloads || 0,
                likes: m.likes || 0,
                pipeline_tag: m.pipeline_tag,
                library_name: m.library_name,
                tags: m.tags
            }));
    } catch  {
        return [];
    }
}
async function fetchModelInfo(modelId) {
    try {
        const response = await fetch(`${HF_API_BASE}/models/${modelId}`, {
            signal: AbortSignal.timeout(TIMEOUT_MS)
        });
        if (!response.ok) return null;
        return await response.json();
    } catch  {
        return null;
    }
}
async function fetchConfigJson(modelId) {
    try {
        const response = await fetch(`${HF_BASE}/${modelId}/resolve/main/config.json`, {
            signal: AbortSignal.timeout(TIMEOUT_MS)
        });
        if (!response.ok) return null;
        return await response.json();
    } catch  {
        return null;
    }
}
async function fetchFullModelSpec(modelId) {
    // Fetch both in parallel
    const [modelInfo, configJson] = await Promise.all([
        fetchModelInfo(modelId),
        fetchConfigJson(modelId)
    ]);
    const warnings = [];
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
    const tc = configJson?.text_config;
    const cfg = configJson;
    const layers = cfg?.num_hidden_layers ?? tc?.num_hidden_layers ?? cfg?.n_layer ?? tc?.n_layer ?? null;
    const kvHeads = cfg?.num_key_value_heads ?? tc?.num_key_value_heads ?? cfg?.n_head_kv ?? tc?.n_head_kv ?? cfg?.num_attention_heads ?? tc?.num_attention_heads ?? null;
    let headDim = cfg?.head_dim ?? tc?.head_dim ?? null;
    if (!headDim) {
        const hs = cfg?.hidden_size ?? tc?.hidden_size;
        const nah = cfg?.num_attention_heads ?? tc?.num_attention_heads;
        if (hs && nah) headDim = Math.floor(hs / nah);
    }
    const maxPositionEmbeddings = cfg?.max_position_embeddings ?? tc?.max_position_embeddings ?? cfg?.max_seq_len ?? tc?.max_seq_len ?? cfg?.model_max_length ?? tc?.model_max_length ?? null;
    if (layers === null) warnings.push('Could not determine layer count.');
    if (kvHeads === null) warnings.push('Could not determine KV head count.');
    if (headDim === null) warnings.push('Could not determine head dimension.');
    const displayName = modelId.split('/').pop() || modelId;
    const spec = {
        name: displayName,
        huggingfaceId: modelId,
        totalParams_B: Math.round(totalParams_B * 100) / 100,
        layers: layers ?? 0,
        kvHeads: kvHeads ?? 0,
        headDim: headDim ?? 0,
        maxPositionEmbeddings: maxPositionEmbeddings ?? undefined,
        description: modelInfo?.config?.architectures?.[0] ? `${modelInfo.config.architectures[0]} · ${modelId}` : modelId
    };
    return {
        spec,
        warnings
    };
}
function formatDownloads(n) {
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
    return n.toString();
}
}),
"[project]/Documents/GitHub/CanIHostIt/src/components/Tooltip.tsx [app-ssr] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>Tooltip
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/Documents/GitHub/CanIHostIt/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/Documents/GitHub/CanIHostIt/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
'use client';
;
;
function Tooltip({ text, children }) {
    const [visible, setVisible] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(false);
    const [position, setPosition] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])('above');
    const triggerRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useRef"])(null);
    const tooltipRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useRef"])(null);
    const tooltipId = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useRef"])(`tooltip-${Math.random().toString(36).slice(2, 8)}`);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        if (visible && triggerRef.current) {
            const rect = triggerRef.current.getBoundingClientRect();
            setPosition(rect.top < 120 ? 'below' : 'above');
        }
    }, [
        visible
    ]);
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
        className: "relative inline-flex items-center",
        children: [
            children,
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                ref: triggerRef,
                type: "button",
                className: "ml-1.5 w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold transition-all duration-200 cursor-help focus:outline-none focus:ring-2",
                style: {
                    background: 'oklch(1 0 0 / 0.06)',
                    color: 'var(--color-text-tertiary)',
                    border: '1px solid oklch(1 0 0 / 0.08)',
                    focusRingColor: 'oklch(0.78 0.15 195 / 0.3)'
                },
                onMouseEnter: ()=>setVisible(true),
                onMouseLeave: ()=>setVisible(false),
                onFocus: ()=>setVisible(true),
                onBlur: ()=>setVisible(false),
                "aria-describedby": tooltipId.current,
                children: "?"
            }, void 0, false, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/Tooltip.tsx",
                lineNumber: 27,
                columnNumber: 7
            }, this),
            visible && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                ref: tooltipRef,
                id: tooltipId.current,
                role: "tooltip",
                className: "absolute z-50 px-3 py-2 text-xs leading-relaxed rounded-lg max-w-[260px] whitespace-normal pointer-events-none animate-fade-in-up",
                style: {
                    background: 'oklch(0.15 0.01 260 / 0.95)',
                    backdropFilter: 'blur(20px)',
                    border: '1px solid oklch(1 0 0 / 0.1)',
                    color: 'var(--color-text-secondary)',
                    boxShadow: '0 8px 32px oklch(0 0 0 / 0.4)',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    ...position === 'above' ? {
                        bottom: 'calc(100% + 8px)'
                    } : {
                        top: 'calc(100% + 8px)'
                    },
                    animationDuration: '0.15s'
                },
                children: text
            }, void 0, false, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/Tooltip.tsx",
                lineNumber: 46,
                columnNumber: 9
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/Tooltip.tsx",
        lineNumber: 25,
        columnNumber: 5
    }, this);
}
}),
"[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx [app-ssr] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>ModelCard
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/Documents/GitHub/CanIHostIt/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/Documents/GitHub/CanIHostIt/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$lib$2f$model$2d$database$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/Documents/GitHub/CanIHostIt/src/lib/model-database.ts [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$components$2f$Tooltip$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/Documents/GitHub/CanIHostIt/src/components/Tooltip.tsx [app-ssr] (ecmascript)");
'use client';
;
;
;
;
function formatGiB(value) {
    if (value >= 1024) return `${(value / 1024).toFixed(1)} TiB`;
    if (value >= 100) return `${value.toFixed(0)} GiB`;
    if (value >= 10) return `${value.toFixed(1)} GiB`;
    return `${value.toFixed(2)} GiB`;
}
function ModelCard({ entry, gpus, results, onUpdate, onRemove, index }) {
    const [searchQuery, setSearchQuery] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])('');
    const [searchResults, setSearchResults] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])([]);
    const [showDropdown, setShowDropdown] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(false);
    const [loading, setLoading] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(false);
    const [fetchWarnings, setFetchWarnings] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])([]);
    const [showAdvanced, setShowAdvanced] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(false);
    const debounceRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useRef"])(null);
    const dropdownRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useRef"])(null);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        const handleClick = (e)=>{
            if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
                setShowDropdown(false);
            }
        };
        document.addEventListener('mousedown', handleClick);
        return ()=>document.removeEventListener('mousedown', handleClick);
    }, []);
    const doSearch = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useCallback"])(async (query)=>{
        if (query.length < 2) {
            setSearchResults([]);
            return;
        }
        const res = await (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$lib$2f$model$2d$database$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["searchHuggingFaceModels"])(query);
        setSearchResults(res);
        setShowDropdown(res.length > 0);
    }, []);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        if (debounceRef.current) clearTimeout(debounceRef.current);
        debounceRef.current = setTimeout(()=>doSearch(searchQuery), 400);
        return ()=>{
            if (debounceRef.current) clearTimeout(debounceRef.current);
        };
    }, [
        searchQuery,
        doSearch
    ]);
    const handleSelectModel = async (modelId)=>{
        setShowDropdown(false);
        setSearchQuery(modelId);
        setLoading(true);
        setFetchWarnings([]);
        const result = await (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$lib$2f$model$2d$database$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["fetchFullModelSpec"])(modelId);
        setLoading(false);
        if (result) {
            setFetchWarnings(result.warnings);
            const maxCtx = result.spec.maxPositionEmbeddings ? Math.min(entry.maxContextTokens, result.spec.maxPositionEmbeddings) : entry.maxContextTokens;
            onUpdate({
                ...entry,
                model: result.spec,
                maxContextTokens: maxCtx
            });
        } else {
            setFetchWarnings([
                'Failed to fetch model info. Check the model ID or enter details manually.'
            ]);
        }
    };
    const updateModel = (patch)=>{
        onUpdate({
            ...entry,
            model: {
                ...entry.model,
                ...patch
            }
        });
    };
    const contextMax = entry.model.maxPositionEmbeddings || 1048576;
    const contextSteps = [
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576
    ].filter((v)=>v <= contextMax);
    if (contextSteps.length === 0) contextSteps.push(contextMax);
    const currentCtxIdx = contextSteps.findIndex((v)=>v >= entry.maxContextTokens);
    const ctxIdx = currentCtxIdx >= 0 ? currentCtxIdx : contextSteps.length - 1;
    const hasBatchOverride = entry.batchSizeOverride !== undefined;
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "glass-card p-5 animate-fade-in-up",
        style: {
            opacity: 0,
            animationDelay: `${index * 0.05}s`
        },
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "flex items-center justify-between mb-4",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex items-center gap-2",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                className: "w-6 h-6 rounded-md flex items-center justify-center text-xs font-bold",
                                style: {
                                    background: 'linear-gradient(135deg, var(--color-accent-cyan), var(--color-accent-violet))',
                                    color: 'white'
                                },
                                children: index + 1
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 103,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("h3", {
                                className: "text-sm font-semibold",
                                style: {
                                    color: 'var(--color-text-primary)'
                                },
                                children: entry.model.name || 'New Model'
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 109,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 102,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                        onClick: ()=>onRemove(entry.id),
                        className: "w-7 h-7 rounded-lg flex items-center justify-center transition-colors",
                        style: {
                            background: 'oklch(0.65 0.22 25 / 0.1)',
                            color: 'var(--color-danger)'
                        },
                        title: "Remove model",
                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("svg", {
                            width: "12",
                            height: "12",
                            viewBox: "0 0 24 24",
                            fill: "none",
                            stroke: "currentColor",
                            strokeWidth: "2",
                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                                d: "M18 6 6 18M6 6l12 12"
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 119,
                                columnNumber: 109
                            }, this)
                        }, void 0, false, {
                            fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                            lineNumber: 119,
                            columnNumber: 11
                        }, this)
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 113,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                lineNumber: 101,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "relative mb-3",
                ref: dropdownRef,
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex items-center gap-2",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "relative flex-1",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                        type: "text",
                                        className: "glass-input",
                                        placeholder: "Search HuggingFace (e.g. meta-llama, Qwen...)",
                                        value: searchQuery,
                                        onChange: (e)=>setSearchQuery(e.target.value),
                                        onFocus: ()=>searchResults.length > 0 && setShowDropdown(true),
                                        style: {
                                            paddingLeft: '32px'
                                        }
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                        lineNumber: 127,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("svg", {
                                        className: "absolute left-3 top-1/2 -translate-y-1/2",
                                        width: "14",
                                        height: "14",
                                        viewBox: "0 0 24 24",
                                        fill: "none",
                                        stroke: "var(--color-text-tertiary)",
                                        strokeWidth: "2",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("circle", {
                                                cx: "11",
                                                cy: "11",
                                                r: "8"
                                            }, void 0, false, {
                                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                                lineNumber: 141,
                                                columnNumber: 15
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                                                d: "m21 21-4.35-4.35"
                                            }, void 0, false, {
                                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                                lineNumber: 141,
                                                columnNumber: 46
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                        lineNumber: 136,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 126,
                                columnNumber: 11
                            }, this),
                            loading && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "w-5 h-5 rounded-full border-2 border-t-transparent animate-spin",
                                style: {
                                    borderColor: 'var(--color-accent-cyan)',
                                    borderTopColor: 'transparent'
                                }
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 145,
                                columnNumber: 13
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 125,
                        columnNumber: 9
                    }, this),
                    showDropdown && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "absolute z-50 top-full mt-1 w-full rounded-xl overflow-hidden",
                        style: {
                            background: 'oklch(0.12 0.01 260 / 0.95)',
                            backdropFilter: 'blur(20px)',
                            border: '1px solid oklch(1 0 0 / 0.08)',
                            boxShadow: '0 12px 40px oklch(0 0 0 / 0.4)',
                            maxHeight: '240px',
                            overflowY: 'auto'
                        },
                        children: searchResults.map((r)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                className: "w-full px-3 py-2.5 text-left transition-colors flex items-center justify-between",
                                style: {
                                    borderBottom: '1px solid oklch(1 0 0 / 0.04)'
                                },
                                onClick: ()=>handleSelectModel(r.id),
                                onMouseOver: (e)=>e.currentTarget.style.background = 'oklch(1 0 0 / 0.04)',
                                onMouseOut: (e)=>e.currentTarget.style.background = 'transparent',
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                                className: "text-sm font-medium",
                                                style: {
                                                    color: 'var(--color-text-primary)'
                                                },
                                                children: r.id
                                            }, void 0, false, {
                                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                                lineNumber: 173,
                                                columnNumber: 19
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                                className: "text-xs",
                                                style: {
                                                    color: 'var(--color-text-tertiary)'
                                                },
                                                children: [
                                                    r.pipeline_tag,
                                                    " · ",
                                                    r.library_name
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                                lineNumber: 174,
                                                columnNumber: 19
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                        lineNumber: 172,
                                        columnNumber: 17
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "text-xs",
                                        style: {
                                            color: 'var(--color-text-tertiary)'
                                        },
                                        children: [
                                            "↓",
                                            (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$lib$2f$model$2d$database$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["formatDownloads"])(r.downloads)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                        lineNumber: 176,
                                        columnNumber: 17
                                    }, this)
                                ]
                            }, r.id, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 164,
                                columnNumber: 15
                            }, this))
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 152,
                        columnNumber: 11
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                lineNumber: 124,
                columnNumber: 7
            }, this),
            fetchWarnings.length > 0 && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "mb-3 px-3 py-2 rounded-lg text-xs",
                style: {
                    background: 'oklch(0.8 0.16 80 / 0.08)',
                    color: 'var(--color-accent-amber)',
                    border: '1px solid oklch(0.8 0.16 80 / 0.15)'
                },
                children: fetchWarnings.map((w, i)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                        children: w
                    }, i, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 188,
                        columnNumber: 40
                    }, this))
            }, void 0, false, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                lineNumber: 187,
                columnNumber: 9
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "grid grid-cols-4 gap-2 mb-3",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                                className: "text-xs mb-0.5 block",
                                style: {
                                    color: 'var(--color-text-tertiary)'
                                },
                                children: "Params (B)"
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 195,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                type: "number",
                                className: "glass-input",
                                value: entry.model.totalParams_B || '',
                                onChange: (e)=>updateModel({
                                        totalParams_B: Number(e.target.value)
                                    }),
                                min: 0.1,
                                step: 0.1
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 196,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 194,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                                className: "text-xs mb-0.5 block",
                                style: {
                                    color: 'var(--color-text-tertiary)'
                                },
                                children: "Layers"
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 199,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                type: "number",
                                className: "glass-input",
                                value: entry.model.layers || '',
                                onChange: (e)=>updateModel({
                                        layers: Number(e.target.value)
                                    }),
                                min: 1
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 200,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 198,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                                className: "text-xs mb-0.5 block",
                                style: {
                                    color: 'var(--color-text-tertiary)'
                                },
                                children: "KV Heads"
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 203,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                type: "number",
                                className: "glass-input",
                                value: entry.model.kvHeads || '',
                                onChange: (e)=>updateModel({
                                        kvHeads: Number(e.target.value)
                                    }),
                                min: 1
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 204,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 202,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                                className: "text-xs mb-0.5 block",
                                style: {
                                    color: 'var(--color-text-tertiary)'
                                },
                                children: "Head Dim"
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 207,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                type: "number",
                                className: "glass-input",
                                value: entry.model.headDim || '',
                                onChange: (e)=>updateModel({
                                        headDim: Number(e.target.value)
                                    }),
                                min: 1
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 208,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 206,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                lineNumber: 193,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "mb-3",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                        className: "text-xs mb-0.5 block",
                        style: {
                            color: 'var(--color-text-tertiary)'
                        },
                        children: "Assigned GPU"
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 214,
                        columnNumber: 9
                    }, this),
                    gpus.length > 0 ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("select", {
                        className: "glass-select",
                        value: entry.gpuId,
                        onChange: (e)=>onUpdate({
                                ...entry,
                                gpuId: e.target.value
                            }),
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("option", {
                                value: "",
                                children: "Select GPU…"
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 221,
                                columnNumber: 13
                            }, this),
                            gpus.map((g)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("option", {
                                    value: g.id,
                                    children: [
                                        g.name,
                                        " (",
                                        g.vramGiB,
                                        " GiB)"
                                    ]
                                }, g.id, true, {
                                    fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                    lineNumber: 223,
                                    columnNumber: 15
                                }, this))
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 216,
                        columnNumber: 11
                    }, this) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                        className: "text-xs py-2",
                        style: {
                            color: 'var(--color-accent-amber)'
                        },
                        children: "Add GPUs to your inventory first"
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 227,
                        columnNumber: 11
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                lineNumber: 213,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "grid grid-cols-2 gap-3 mb-3",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$components$2f$Tooltip$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["default"], {
                                text: "FP8 uses 1 byte per parameter (faster, less VRAM). BF16 uses 2 bytes (higher precision, more VRAM).",
                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                                    className: "text-xs font-medium uppercase tracking-wider",
                                    style: {
                                        color: 'var(--color-text-secondary)'
                                    },
                                    children: "Weight Precision"
                                }, void 0, false, {
                                    fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                    lineNumber: 237,
                                    columnNumber: 13
                                }, this)
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 236,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "toggle-group mt-1",
                                children: [
                                    'FP8',
                                    'BF16'
                                ].map((q)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                        className: `toggle-option ${entry.quantization === q ? 'active' : ''}`,
                                        onClick: ()=>onUpdate({
                                                ...entry,
                                                quantization: q
                                            }),
                                        children: q
                                    }, q, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                        lineNumber: 243,
                                        columnNumber: 15
                                    }, this))
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 241,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 235,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$components$2f$Tooltip$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["default"], {
                                text: "KV cache precision. FP8 (1 byte) halves KV memory vs BF16 (2 bytes), doubling batch size capacity.",
                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                                    className: "text-xs font-medium uppercase tracking-wider",
                                    style: {
                                        color: 'var(--color-text-secondary)'
                                    },
                                    children: "KV Cache Type"
                                }, void 0, false, {
                                    fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                    lineNumber: 251,
                                    columnNumber: 13
                                }, this)
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 250,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "toggle-group mt-1",
                                children: [
                                    'FP8',
                                    'BF16'
                                ].map((q)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                        className: `toggle-option ${entry.kvCacheType === q ? 'active' : ''}`,
                                        onClick: ()=>onUpdate({
                                                ...entry,
                                                kvCacheType: q
                                            }),
                                        children: q
                                    }, q, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                        lineNumber: 257,
                                        columnNumber: 15
                                    }, this))
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 255,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 249,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                lineNumber: 234,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "mb-3",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$components$2f$Tooltip$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["default"], {
                        text: `Maximum tokens per request. ${entry.model.maxPositionEmbeddings ? `This model supports up to ${(entry.model.maxPositionEmbeddings / 1024).toFixed(0)}K tokens.` : 'Larger context = more KV cache = lower batch size.'}`,
                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                            className: "text-xs font-medium uppercase tracking-wider",
                            style: {
                                color: 'var(--color-text-secondary)'
                            },
                            children: "Max Context Tokens"
                        }, void 0, false, {
                            fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                            lineNumber: 268,
                            columnNumber: 11
                        }, this)
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 267,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                        type: "range",
                        className: "glass-slider mt-1",
                        min: 0,
                        max: contextSteps.length - 1,
                        step: 1,
                        value: ctxIdx,
                        onChange: (e)=>onUpdate({
                                ...entry,
                                maxContextTokens: contextSteps[Number(e.target.value)]
                            })
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 272,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex justify-between text-xs mt-0.5",
                        style: {
                            color: 'var(--color-text-tertiary)'
                        },
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                children: [
                                    (contextSteps[0] / 1024).toFixed(0),
                                    "K"
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 278,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                className: "font-medium",
                                style: {
                                    color: 'var(--color-accent-cyan)'
                                },
                                children: entry.maxContextTokens >= 1048576 ? `${(entry.maxContextTokens / 1048576).toFixed(0)}M` : `${(entry.maxContextTokens / 1024).toFixed(0)}K`
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 279,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                children: [
                                    (contextSteps[contextSteps.length - 1] / 1024).toFixed(0),
                                    "K"
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 282,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 277,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                lineNumber: 266,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "mb-3",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$components$2f$Tooltip$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["default"], {
                        text: "Total simultaneous requests across all replicas for this model. The system auto-provisions enough replicas based on the optimal batch size.",
                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                            className: "text-xs font-medium uppercase tracking-wider",
                            style: {
                                color: 'var(--color-text-secondary)'
                            },
                            children: "Target Concurrency"
                        }, void 0, false, {
                            fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                            lineNumber: 289,
                            columnNumber: 11
                        }, this)
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 288,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex items-center gap-2 mt-1",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                type: "range",
                                className: "glass-slider flex-1",
                                min: 1,
                                max: 1000,
                                value: entry.targetConcurrency,
                                onChange: (e)=>onUpdate({
                                        ...entry,
                                        targetConcurrency: Number(e.target.value)
                                    })
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 294,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                type: "number",
                                className: "glass-input",
                                value: entry.targetConcurrency,
                                onChange: (e)=>onUpdate({
                                        ...entry,
                                        targetConcurrency: Math.max(1, Number(e.target.value))
                                    }),
                                min: 1,
                                max: 10000,
                                style: {
                                    width: '64px',
                                    flex: 'none',
                                    textAlign: 'center'
                                }
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 295,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 293,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                lineNumber: 287,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                className: "text-xs mb-2 transition-colors",
                style: {
                    color: 'var(--color-text-tertiary)'
                },
                onClick: ()=>setShowAdvanced(!showAdvanced),
                children: [
                    showAdvanced ? '▾' : '▸',
                    " Advanced Latency Tuning"
                ]
            }, void 0, true, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                lineNumber: 300,
                columnNumber: 7
            }, this),
            showAdvanced && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "mb-3 p-3 rounded-lg",
                style: {
                    background: 'oklch(1 0 0 / 0.02)',
                    border: '1px solid oklch(1 0 0 / 0.04)'
                },
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex items-center justify-between mb-2",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$components$2f$Tooltip$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["default"], {
                                text: "Override the auto-calculated batch size. Use a lower value to reduce time-to-first-token (latency) at the cost of more replicas. Leave unchecked for optimal throughput.",
                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                                    className: "text-xs font-medium uppercase tracking-wider",
                                    style: {
                                        color: 'var(--color-text-secondary)'
                                    },
                                    children: "Manual Batch Override"
                                }, void 0, false, {
                                    fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                    lineNumber: 312,
                                    columnNumber: 15
                                }, this)
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 311,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                                className: "flex items-center gap-2 cursor-pointer",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                        type: "checkbox",
                                        checked: hasBatchOverride,
                                        onChange: (e)=>{
                                            if (e.target.checked) {
                                                onUpdate({
                                                    ...entry,
                                                    batchSizeOverride: results?.optimalBatchSize ?? 1
                                                });
                                            } else {
                                                const { batchSizeOverride: _, ...rest } = entry;
                                                onUpdate(rest);
                                            }
                                        },
                                        className: "w-3.5 h-3.5 rounded"
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                        lineNumber: 317,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "text-xs",
                                        style: {
                                            color: 'var(--color-text-tertiary)'
                                        },
                                        children: "Enable"
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                        lineNumber: 330,
                                        columnNumber: 15
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 316,
                                columnNumber: 13
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 310,
                        columnNumber: 11
                    }, this),
                    hasBatchOverride && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex items-center gap-2",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                type: "range",
                                className: "glass-slider flex-1",
                                min: 1,
                                max: results?.optimalBatchSize ?? 64,
                                step: 1,
                                value: entry.batchSizeOverride || 1,
                                onChange: (e)=>onUpdate({
                                        ...entry,
                                        batchSizeOverride: Number(e.target.value)
                                    })
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 335,
                                columnNumber: 15
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                type: "number",
                                className: "glass-input",
                                value: entry.batchSizeOverride || 1,
                                onChange: (e)=>onUpdate({
                                        ...entry,
                                        batchSizeOverride: Math.max(1, Number(e.target.value))
                                    }),
                                min: 1,
                                max: results?.optimalBatchSize ?? 9999,
                                style: {
                                    width: '64px',
                                    flex: 'none',
                                    textAlign: 'center'
                                }
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 341,
                                columnNumber: 15
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                className: "text-xs",
                                style: {
                                    color: 'var(--color-text-tertiary)'
                                },
                                children: [
                                    "/ ",
                                    results?.optimalBatchSize ?? '—'
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 348,
                                columnNumber: 15
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 334,
                        columnNumber: 13
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                lineNumber: 309,
                columnNumber: 9
            }, this),
            results && entry.gpuId && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "mt-3 pt-3",
                style: {
                    borderTop: '1px solid oklch(1 0 0 / 0.06)'
                },
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "grid grid-cols-4 gap-2 text-center",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                        className: "text-xs",
                                        style: {
                                            color: 'var(--color-text-tertiary)'
                                        },
                                        children: "Optimal Batch"
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                        lineNumber: 364,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                        className: "text-sm font-bold",
                                        style: {
                                            color: 'var(--color-accent-emerald)'
                                        },
                                        children: [
                                            results.optimalBatchSize,
                                            hasBatchOverride && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "text-xs font-normal ml-1",
                                                style: {
                                                    color: 'var(--color-accent-amber)'
                                                },
                                                children: [
                                                    "→",
                                                    results.effectiveBatchSize
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                                lineNumber: 368,
                                                columnNumber: 19
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                        lineNumber: 365,
                                        columnNumber: 15
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 363,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                        className: "text-xs",
                                        style: {
                                            color: 'var(--color-text-tertiary)'
                                        },
                                        children: "TP×PP"
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                        lineNumber: 373,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                        className: "text-sm font-bold",
                                        style: {
                                            color: 'var(--color-accent-cyan)'
                                        },
                                        children: [
                                            results.tpSize,
                                            "×",
                                            results.ppSize
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                        lineNumber: 374,
                                        columnNumber: 15
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 372,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                        className: "text-xs",
                                        style: {
                                            color: 'var(--color-text-tertiary)'
                                        },
                                        children: "Replicas"
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                        lineNumber: 377,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                        className: "text-sm font-bold",
                                        style: {
                                            color: 'var(--color-text-primary)'
                                        },
                                        children: results.replicas
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                        lineNumber: 378,
                                        columnNumber: 15
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 376,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                        className: "text-xs",
                                        style: {
                                            color: 'var(--color-text-tertiary)'
                                        },
                                        children: "GPUs"
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                        lineNumber: 381,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                        className: "text-sm font-bold",
                                        style: {
                                            color: 'var(--color-text-primary)'
                                        },
                                        children: results.totalGpus
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                        lineNumber: 382,
                                        columnNumber: 15
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 380,
                                columnNumber: 13
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 362,
                        columnNumber: 11
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "mt-2 flex items-center justify-between text-xs",
                        style: {
                            color: 'var(--color-text-tertiary)'
                        },
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                children: [
                                    "Weights: ",
                                    formatGiB(results.totalWeightsGiB)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 387,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                children: [
                                    "KV free: ",
                                    formatGiB(results.vramLeftForKvGiB)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 388,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                children: [
                                    "KV/user: ",
                                    formatGiB(results.kvCachePerUserGiB)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                                lineNumber: 389,
                                columnNumber: 13
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                        lineNumber: 386,
                        columnNumber: 11
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
                lineNumber: 358,
                columnNumber: 9
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx",
        lineNumber: 96,
        columnNumber: 5
    }, this);
}
}),
"[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx [app-ssr] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>ResultsPanel
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/Documents/GitHub/CanIHostIt/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
'use client';
;
function formatGiB(value) {
    if (value >= 1024) return `${(value / 1024).toFixed(1)} TiB`;
    if (value >= 100) return `${value.toFixed(0)} GiB`;
    if (value >= 10) return `${value.toFixed(1)} GiB`;
    return `${value.toFixed(2)} GiB`;
}
function MetricCard({ label, value, unit, accentClass, subtitle, stagger }) {
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: `metric-card animate-fade-in-up ${accentClass || ''} ${stagger ? `stagger-${stagger}` : ''}`,
        style: {
            opacity: 0
        },
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                className: "text-xs font-medium uppercase tracking-wider mb-2",
                style: {
                    color: 'var(--color-text-tertiary)'
                },
                children: label
            }, void 0, false, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                lineNumber: 24,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "flex items-baseline gap-1.5",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                        className: "text-2xl font-bold tracking-tight",
                        style: {
                            color: 'var(--color-text-primary)'
                        },
                        children: value
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                        lineNumber: 26,
                        columnNumber: 9
                    }, this),
                    unit && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                        className: "text-sm font-medium",
                        style: {
                            color: 'var(--color-text-secondary)'
                        },
                        children: unit
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                        lineNumber: 27,
                        columnNumber: 18
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                lineNumber: 25,
                columnNumber: 7
            }, this),
            subtitle && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                className: "text-xs mt-1.5",
                style: {
                    color: 'var(--color-text-tertiary)'
                },
                children: subtitle
            }, void 0, false, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                lineNumber: 29,
                columnNumber: 20
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
        lineNumber: 23,
        columnNumber: 5
    }, this);
}
function ModelBreakdownRow({ r }) {
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "flex items-center justify-between py-2.5",
        style: {
            borderBottom: '1px solid oklch(1 0 0 / 0.04)'
        },
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "flex-1 min-w-0",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                        className: "text-sm font-medium truncate",
                        style: {
                            color: 'var(--color-text-primary)'
                        },
                        children: r.modelName
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                        lineNumber: 38,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                        className: "text-xs",
                        style: {
                            color: 'var(--color-text-tertiary)'
                        },
                        children: r.gpuName
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                        lineNumber: 39,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                lineNumber: 37,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "flex gap-4 text-xs text-right",
                style: {
                    color: 'var(--color-text-secondary)'
                },
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                style: {
                                    color: 'var(--color-text-tertiary)'
                                },
                                children: "Batch"
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                                lineNumber: 43,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                className: "font-medium",
                                style: {
                                    color: 'var(--color-accent-emerald)'
                                },
                                children: [
                                    r.effectiveBatchSize,
                                    r.effectiveBatchSize !== r.optimalBatchSize ? '*' : ''
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                                lineNumber: 44,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                        lineNumber: 42,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                style: {
                                    color: 'var(--color-text-tertiary)'
                                },
                                children: "TP×PP"
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                                lineNumber: 47,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                className: "font-medium",
                                style: {
                                    color: 'var(--color-accent-cyan)'
                                },
                                children: [
                                    r.tpSize,
                                    "×",
                                    r.ppSize
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                                lineNumber: 48,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                        lineNumber: 46,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                style: {
                                    color: 'var(--color-text-tertiary)'
                                },
                                children: "Replicas"
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                                lineNumber: 51,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                className: "font-medium",
                                children: r.replicas
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                                lineNumber: 52,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                        lineNumber: 50,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                style: {
                                    color: 'var(--color-text-tertiary)'
                                },
                                children: "GPUs"
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                                lineNumber: 55,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                className: "font-bold",
                                children: r.totalGpus
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                                lineNumber: 56,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                        lineNumber: 54,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                style: {
                                    color: 'var(--color-text-tertiary)'
                                },
                                children: "VRAM"
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                                lineNumber: 59,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                className: "font-medium",
                                children: formatGiB(r.totalVramGiB)
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                                lineNumber: 60,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                        lineNumber: 58,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                lineNumber: 41,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
        lineNumber: 36,
        columnNumber: 5
    }, this);
}
function ResultsPanel({ fleet, rackPowerBudgetKw, nodePowerKw }) {
    const hasModels = fleet.modelResults.length > 0;
    if (!hasModels) {
        return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
            className: "glass-card p-8 flex flex-col items-center justify-center gap-3 min-h-[300px]",
            children: [
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    className: "w-12 h-12 rounded-2xl flex items-center justify-center",
                    style: {
                        background: 'oklch(1 0 0 / 0.04)'
                    },
                    children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("svg", {
                        width: "24",
                        height: "24",
                        viewBox: "0 0 24 24",
                        fill: "none",
                        stroke: "var(--color-text-tertiary)",
                        strokeWidth: "1.5",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("rect", {
                                x: "4",
                                y: "4",
                                width: "6",
                                height: "6",
                                rx: "1"
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                                lineNumber: 75,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("rect", {
                                x: "14",
                                y: "4",
                                width: "6",
                                height: "6",
                                rx: "1"
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                                lineNumber: 75,
                                columnNumber: 60
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("rect", {
                                x: "4",
                                y: "14",
                                width: "6",
                                height: "6",
                                rx: "1"
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                                lineNumber: 76,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("rect", {
                                x: "14",
                                y: "14",
                                width: "6",
                                height: "6",
                                rx: "1"
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                                lineNumber: 76,
                                columnNumber: 61
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                        lineNumber: 74,
                        columnNumber: 11
                    }, this)
                }, void 0, false, {
                    fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                    lineNumber: 73,
                    columnNumber: 9
                }, this),
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                    className: "text-sm",
                    style: {
                        color: 'var(--color-text-tertiary)'
                    },
                    children: "Add models and assign GPUs to see infrastructure requirements"
                }, void 0, false, {
                    fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                    lineNumber: 79,
                    columnNumber: 9
                }, this)
            ]
        }, void 0, true, {
            fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
            lineNumber: 72,
            columnNumber: 7
        }, this);
    }
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "flex flex-col gap-4",
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("h2", {
                        className: "text-lg font-semibold tracking-tight",
                        style: {
                            color: 'var(--color-text-primary)'
                        },
                        children: "Fleet Requirements"
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                        lineNumber: 89,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                        className: "text-xs mt-1",
                        style: {
                            color: 'var(--color-text-tertiary)'
                        },
                        children: [
                            fleet.modelResults.length,
                            " model",
                            fleet.modelResults.length !== 1 ? 's' : '',
                            " · ",
                            rackPowerBudgetKw,
                            " kW/rack · ",
                            nodePowerKw,
                            " kW/node"
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                        lineNumber: 92,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                lineNumber: 88,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "grid grid-cols-2 lg:grid-cols-4 gap-3",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(MetricCard, {
                        label: "Total VRAM",
                        value: formatGiB(fleet.totalVramGiB),
                        accentClass: "glow-cyan",
                        subtitle: `${fleet.totalGpus} GPUs total`,
                        stagger: 1
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                        lineNumber: 99,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(MetricCard, {
                        label: "Total Nodes",
                        value: fleet.totalNodes.toString(),
                        unit: "nodes",
                        accentClass: "glow-violet",
                        stagger: 2
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                        lineNumber: 100,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(MetricCard, {
                        label: "Total GPUs",
                        value: fleet.totalGpus.toString(),
                        unit: "GPUs",
                        accentClass: "glow-emerald",
                        stagger: 3
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                        lineNumber: 101,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(MetricCard, {
                        label: "Power",
                        value: fleet.totalPowerKw.toFixed(0),
                        unit: "kW",
                        accentClass: "glow-amber",
                        subtitle: `${fleet.totalRacks} rack${fleet.totalRacks !== 1 ? 's' : ''}`,
                        stagger: 4
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                        lineNumber: 102,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                lineNumber: 98,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "glass-card p-5 animate-fade-in-up stagger-5",
                style: {
                    opacity: 0
                },
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("h3", {
                        className: "text-xs font-medium uppercase tracking-wider mb-3",
                        style: {
                            color: 'var(--color-text-secondary)'
                        },
                        children: "Per-Model Breakdown"
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                        lineNumber: 107,
                        columnNumber: 9
                    }, this),
                    fleet.modelResults.map((r)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(ModelBreakdownRow, {
                            r: r
                        }, r.entryId, false, {
                            fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                            lineNumber: 111,
                            columnNumber: 11
                        }, this))
                ]
            }, void 0, true, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
                lineNumber: 106,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx",
        lineNumber: 87,
        columnNumber: 5
    }, this);
}
}),
"[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx [app-ssr] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>Home
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/Documents/GitHub/CanIHostIt/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/Documents/GitHub/CanIHostIt/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$lib$2f$calculator$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/Documents/GitHub/CanIHostIt/src/lib/calculator.ts [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$lib$2f$gpu$2d$database$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/Documents/GitHub/CanIHostIt/src/lib/gpu-database.ts [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$components$2f$GpuInventory$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/Documents/GitHub/CanIHostIt/src/components/GpuInventory.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$components$2f$ModelCard$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/Documents/GitHub/CanIHostIt/src/components/ModelCard.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$components$2f$ResultsPanel$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/Documents/GitHub/CanIHostIt/src/components/ResultsPanel.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$components$2f$Tooltip$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/Documents/GitHub/CanIHostIt/src/components/Tooltip.tsx [app-ssr] (ecmascript)");
'use client';
;
;
;
;
;
;
;
;
const EMPTY_MODEL = {
    name: 'New Model',
    totalParams_B: 0,
    layers: 0,
    kvHeads: 0,
    headDim: 0
};
function createModelEntry(gpuId = '') {
    return {
        id: `model_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
        model: {
            ...EMPTY_MODEL
        },
        gpuId,
        quantization: 'FP8',
        kvCacheType: 'FP8',
        maxContextTokens: 131072,
        targetConcurrency: 64
    };
}
function Home() {
    // ─── State ──────────────────────────────────────────────
    const [gpuInventory, setGpuInventory] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])([]);
    const [modelEntries, setModelEntries] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])([]);
    const [rackPowerBudgetKw, setRackPowerBudgetKw] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(20);
    const [nodePowerKw, setNodePowerKw] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(10);
    const [gpuLoaded, setGpuLoaded] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(false);
    const [activeTab, setActiveTab] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])('models');
    // ─── Load GPU inventory from localStorage ───────────────
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        const saved = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$lib$2f$gpu$2d$database$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["loadGpuInventory"])();
        setGpuInventory(saved);
        setGpuLoaded(true);
    }, []);
    // ─── Save GPU inventory to localStorage ─────────────────
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        if (gpuLoaded) {
            (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$lib$2f$gpu$2d$database$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["saveGpuInventory"])(gpuInventory);
        }
    }, [
        gpuInventory,
        gpuLoaded
    ]);
    // ─── GPU CRUD ───────────────────────────────────────────
    const addGpu = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useCallback"])((gpu)=>setGpuInventory((prev)=>[
                ...prev,
                gpu
            ]), []);
    const removeGpu = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useCallback"])((id)=>setGpuInventory((prev)=>prev.filter((g)=>g.id !== id)), []);
    const updateGpu = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useCallback"])((gpu)=>setGpuInventory((prev)=>prev.map((g)=>g.id === gpu.id ? gpu : g)), []);
    // ─── Model CRUD ─────────────────────────────────────────
    const addModel = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useCallback"])(()=>{
        const defaultGpuId = gpuInventory.length > 0 ? gpuInventory[0].id : '';
        setModelEntries((prev)=>[
                ...prev,
                createModelEntry(defaultGpuId)
            ]);
        setActiveTab('models');
    }, [
        gpuInventory
    ]);
    const removeModel = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useCallback"])((id)=>setModelEntries((prev)=>prev.filter((m)=>m.id !== id)), []);
    const updateModel = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useCallback"])((entry)=>setModelEntries((prev)=>prev.map((m)=>m.id === entry.id ? entry : m)), []);
    // ─── Calculation ────────────────────────────────────────
    const fleet = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useMemo"])(()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$lib$2f$calculator$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["calculateFleetTotals"])(modelEntries, gpuInventory, rackPowerBudgetKw, nodePowerKw), [
        modelEntries,
        gpuInventory,
        rackPowerBudgetKw,
        nodePowerKw
    ]);
    // Build a results map for inline display
    const resultsMap = (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useMemo"])(()=>{
        const map = new Map();
        fleet.modelResults.forEach((r)=>map.set(r.entryId, r));
        return map;
    }, [
        fleet.modelResults
    ]);
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "relative z-10",
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("header", {
                className: "px-7 py-5 flex items-center justify-between",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex items-center gap-3",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "w-8 h-8 rounded-lg flex items-center justify-center",
                                style: {
                                    background: 'linear-gradient(135deg, var(--color-accent-cyan), var(--color-accent-violet))'
                                },
                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("svg", {
                                    width: "18",
                                    height: "18",
                                    viewBox: "0 0 24 24",
                                    fill: "none",
                                    stroke: "white",
                                    strokeWidth: "2",
                                    strokeLinecap: "round",
                                    strokeLinejoin: "round",
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("rect", {
                                            x: "4",
                                            y: "4",
                                            width: "6",
                                            height: "6",
                                            rx: "1"
                                        }, void 0, false, {
                                            fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                            lineNumber: 93,
                                            columnNumber: 15
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("rect", {
                                            x: "14",
                                            y: "4",
                                            width: "6",
                                            height: "6",
                                            rx: "1"
                                        }, void 0, false, {
                                            fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                            lineNumber: 93,
                                            columnNumber: 62
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("rect", {
                                            x: "4",
                                            y: "14",
                                            width: "6",
                                            height: "6",
                                            rx: "1"
                                        }, void 0, false, {
                                            fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                            lineNumber: 94,
                                            columnNumber: 15
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("rect", {
                                            x: "14",
                                            y: "14",
                                            width: "6",
                                            height: "6",
                                            rx: "1"
                                        }, void 0, false, {
                                            fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                            lineNumber: 94,
                                            columnNumber: 63
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                    lineNumber: 92,
                                    columnNumber: 13
                                }, this)
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                lineNumber: 88,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("h1", {
                                        className: "text-base font-bold tracking-tight gradient-text",
                                        children: "CanIHostIt"
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                        lineNumber: 98,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                        className: "text-xs",
                                        style: {
                                            color: 'var(--color-text-tertiary)'
                                        },
                                        children: "AI Infrastructure Capacity Planner"
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                        lineNumber: 99,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                lineNumber: 97,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                        lineNumber: 87,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex items-center gap-2",
                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                            onClick: addModel,
                            className: "text-xs px-4 py-2 rounded-lg font-medium transition-all",
                            style: {
                                background: 'linear-gradient(135deg, oklch(0.78 0.15 195 / 0.15), oklch(0.65 0.2 290 / 0.15))',
                                color: 'var(--color-accent-cyan)',
                                border: '1px solid oklch(0.78 0.15 195 / 0.2)'
                            },
                            children: "+ Add Model"
                        }, void 0, false, {
                            fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                            lineNumber: 103,
                            columnNumber: 11
                        }, this)
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                        lineNumber: 102,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                lineNumber: 86,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "dashboard-grid",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex flex-col gap-4",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "toggle-group",
                                children: [
                                    'models',
                                    'hardware',
                                    'settings'
                                ].map((tab)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                        className: `toggle-option ${activeTab === tab ? 'active' : ''}`,
                                        onClick: ()=>setActiveTab(tab),
                                        children: tab === 'models' ? `Models (${modelEntries.length})` : tab === 'hardware' ? `GPUs (${gpuInventory.length})` : 'Settings'
                                    }, tab, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                        lineNumber: 124,
                                        columnNumber: 15
                                    }, this))
                            }, void 0, false, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                lineNumber: 122,
                                columnNumber: 11
                            }, this),
                            activeTab === 'models' && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex flex-col gap-3",
                                children: [
                                    modelEntries.length === 0 ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "glass-card p-6 flex flex-col items-center gap-3",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                                className: "text-sm",
                                                style: {
                                                    color: 'var(--color-text-tertiary)'
                                                },
                                                children: "No models configured"
                                            }, void 0, false, {
                                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                                lineNumber: 139,
                                                columnNumber: 19
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                onClick: addModel,
                                                className: "text-xs px-4 py-2 rounded-lg transition-all",
                                                style: {
                                                    background: 'linear-gradient(135deg, oklch(0.78 0.15 195 / 0.15), oklch(0.65 0.2 290 / 0.15))',
                                                    color: 'var(--color-accent-cyan)',
                                                    border: '1px solid oklch(0.78 0.15 195 / 0.2)'
                                                },
                                                children: "+ Add Your First Model"
                                            }, void 0, false, {
                                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                                lineNumber: 140,
                                                columnNumber: 19
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                        lineNumber: 138,
                                        columnNumber: 17
                                    }, this) : modelEntries.map((entry, i)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$components$2f$ModelCard$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["default"], {
                                            entry: entry,
                                            gpus: gpuInventory,
                                            results: resultsMap.get(entry.id) || null,
                                            onUpdate: updateModel,
                                            onRemove: removeModel,
                                            index: i
                                        }, entry.id, false, {
                                            fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                            lineNumber: 154,
                                            columnNumber: 19
                                        }, this)),
                                    modelEntries.length > 0 && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                        onClick: addModel,
                                        className: "text-xs px-3 py-2.5 rounded-lg transition-all",
                                        style: {
                                            background: 'oklch(1 0 0 / 0.03)',
                                            color: 'var(--color-text-secondary)',
                                            border: '1px dashed oklch(1 0 0 / 0.1)'
                                        },
                                        children: "+ Add Another Model"
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                        lineNumber: 166,
                                        columnNumber: 17
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                lineNumber: 136,
                                columnNumber: 13
                            }, this),
                            activeTab === 'hardware' && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "glass-card p-5",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("h2", {
                                        className: "text-sm font-semibold mb-3",
                                        style: {
                                            color: 'var(--color-text-primary)'
                                        },
                                        children: "GPU Inventory"
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                        lineNumber: 184,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                        className: "text-xs mb-4",
                                        style: {
                                            color: 'var(--color-text-tertiary)'
                                        },
                                        children: "Add GPUs to your inventory. Models are assigned to GPUs from this list."
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                        lineNumber: 185,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$components$2f$GpuInventory$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["default"], {
                                        gpus: gpuInventory,
                                        onAdd: addGpu,
                                        onRemove: removeGpu,
                                        onUpdate: updateGpu
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                        lineNumber: 188,
                                        columnNumber: 15
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                lineNumber: 183,
                                columnNumber: 13
                            }, this),
                            activeTab === 'settings' && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "glass-card p-5 flex flex-col gap-4",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("h2", {
                                        className: "text-sm font-semibold",
                                        style: {
                                            color: 'var(--color-text-primary)'
                                        },
                                        children: "Infrastructure Settings"
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                        lineNumber: 195,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$components$2f$Tooltip$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["default"], {
                                                text: "Power capacity per datacenter rack in kilowatts. Determines how many GPU nodes fit per rack.",
                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                                                    className: "text-xs font-medium uppercase tracking-wider",
                                                    style: {
                                                        color: 'var(--color-text-secondary)'
                                                    },
                                                    children: "Rack Power Budget"
                                                }, void 0, false, {
                                                    fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                                    lineNumber: 199,
                                                    columnNumber: 19
                                                }, this)
                                            }, void 0, false, {
                                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                                lineNumber: 198,
                                                columnNumber: 17
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                className: "flex items-center gap-2 mt-1",
                                                children: [
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                                        type: "number",
                                                        className: "glass-input",
                                                        value: rackPowerBudgetKw,
                                                        onChange: (e)=>setRackPowerBudgetKw(Math.max(1, Number(e.target.value))),
                                                        min: 1,
                                                        style: {
                                                            width: '100px'
                                                        }
                                                    }, void 0, false, {
                                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                                        lineNumber: 204,
                                                        columnNumber: 19
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                        className: "text-xs",
                                                        style: {
                                                            color: 'var(--color-text-tertiary)'
                                                        },
                                                        children: "kW per rack"
                                                    }, void 0, false, {
                                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                                        lineNumber: 212,
                                                        columnNumber: 19
                                                    }, this)
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                                lineNumber: 203,
                                                columnNumber: 17
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                        lineNumber: 197,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$components$2f$Tooltip$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["default"], {
                                                text: "Power consumption per GPU node (server). Used to calculate how many nodes fit within the rack power budget.",
                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                                                    className: "text-xs font-medium uppercase tracking-wider",
                                                    style: {
                                                        color: 'var(--color-text-secondary)'
                                                    },
                                                    children: "Node Power Draw"
                                                }, void 0, false, {
                                                    fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                                    lineNumber: 218,
                                                    columnNumber: 19
                                                }, this)
                                            }, void 0, false, {
                                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                                lineNumber: 217,
                                                columnNumber: 17
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                className: "flex items-center gap-2 mt-1",
                                                children: [
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                                        type: "number",
                                                        className: "glass-input",
                                                        value: nodePowerKw,
                                                        onChange: (e)=>setNodePowerKw(Math.max(0.5, Number(e.target.value))),
                                                        min: 0.5,
                                                        step: 0.5,
                                                        style: {
                                                            width: '100px'
                                                        }
                                                    }, void 0, false, {
                                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                                        lineNumber: 223,
                                                        columnNumber: 19
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                        className: "text-xs",
                                                        style: {
                                                            color: 'var(--color-text-tertiary)'
                                                        },
                                                        children: "kW per node"
                                                    }, void 0, false, {
                                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                                        lineNumber: 232,
                                                        columnNumber: 19
                                                    }, this)
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                                lineNumber: 222,
                                                columnNumber: 17
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                        lineNumber: 216,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "pt-2",
                                        style: {
                                            borderTop: '1px solid oklch(1 0 0 / 0.06)'
                                        },
                                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                            className: "text-xs",
                                            style: {
                                                color: 'var(--color-text-tertiary)'
                                            },
                                            children: [
                                                "Nodes per rack: ",
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("strong", {
                                                    className: "gradient-text",
                                                    children: nodePowerKw > 0 ? Math.floor(rackPowerBudgetKw / nodePowerKw) : '∞'
                                                }, void 0, false, {
                                                    fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                                    lineNumber: 238,
                                                    columnNumber: 35
                                                }, this)
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                            lineNumber: 237,
                                            columnNumber: 17
                                        }, this)
                                    }, void 0, false, {
                                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                        lineNumber: 236,
                                        columnNumber: 15
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                                lineNumber: 194,
                                columnNumber: 13
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                        lineNumber: 120,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$src$2f$components$2f$ResultsPanel$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["default"], {
                        fleet: fleet,
                        rackPowerBudgetKw: rackPowerBudgetKw,
                        nodePowerKw: nodePowerKw
                    }, void 0, false, {
                        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                        lineNumber: 246,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                lineNumber: 118,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("footer", {
                className: "px-7 py-4 text-center",
                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$Documents$2f$GitHub$2f$CanIHostIt$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                    className: "text-xs",
                    style: {
                        color: 'var(--color-text-tertiary)'
                    },
                    children: "Batch size auto-optimized from leftover VRAM. 20% vLLM framework overhead. GPU utilization defaults to 90%."
                }, void 0, false, {
                    fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                    lineNumber: 251,
                    columnNumber: 9
                }, this)
            }, void 0, false, {
                fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
                lineNumber: 250,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/Documents/GitHub/CanIHostIt/src/app/page.tsx",
        lineNumber: 84,
        columnNumber: 5
    }, this);
}
}),
];

//# sourceMappingURL=Documents_GitHub_CanIHostIt_src_0z5wzcd._.js.map