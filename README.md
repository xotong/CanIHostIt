# CanIHostIt

CanIHostIt is a **web-based AI infrastructure capacity planner** for estimating the GPU, node, rack, and power requirements needed to serve one or more models.

It helps answer: **“Can I host this model (or fleet of models) on my available hardware?”**

## What it does

- Configure one or more model deployments
- Assign each model to a GPU type from your inventory
- Automatically calculate:
  - Weight memory (including framework overhead)
  - KV-cache memory per user
  - TP/PP split
  - Optimal batch size
  - Replicas required for target concurrency
  - Total GPUs, nodes, racks, and power draw
- View outputs in:
  - **Summary view** (compact per-model metrics)
  - **Table view** (detailed, step-by-step calculations)

## Key assumptions & methodology

The calculator uses these core assumptions:

- **Weight precision:** FP8 = 1 byte/param, BF16 = 2 bytes/param
- **KV cache precision:** FP8 = 1 byte/element, BF16 = 2 bytes/element
- **Framework overhead:** 20% added to base weight memory
- **Usable VRAM:** GPU VRAM × configured utilization (default 90%)

High-level flow:

1. Compute base model weights from parameter count and precision
2. Add 20% framework overhead
3. Choose TP/PP based on weight fit (KV not used for TP/PP selection)
4. Use remaining VRAM for KV cache and derive optimal batch size
5. Compute replicas from target concurrency
6. Aggregate fleet totals (GPUs, nodes, racks, power)

## Tech stack

- Next.js 16
- React 19
- TypeScript
- Tailwind CSS

## Getting started

### Prerequisites

- Node.js 20+ (recommended)
- npm

### Install dependencies

```bash
npm install
```

### Run in development

```bash
npm run dev
```

Then open: <http://localhost:3000>

### Build for production

```bash
npm run build
```

### Start production server

```bash
npm run start
```

### Lint

```bash
npm run lint
```

## Available scripts

From `package.json`:

- `npm run dev` — start Next.js dev server (Turbopack)
- `npm run build` — create production build
- `npm run start` — run production server
- `npm run lint` — run Next.js lint checks

## Notes

- GPU inventory is persisted in browser local storage.
- The app can fetch model metadata from Hugging Face to prefill architecture fields.
- There is currently no dedicated automated test suite configured in this repository.
