# k-hop Diffusion Co-Association Loss — design rationale

A differentiable, scalable training objective for affinity-based instance
segmentation. This deck explains **why k-hop** was chosen over per-edge losses
(MALIS) and the non-differentiable greedy solver (Mutex Watershed).

Context: nanoCosmos · Cosmos3-Edge affinity head · mutex-watershed at inference.

> **Thesis.** Instance identity is a *transitive* property, not a per-edge one.
> So supervise *multi-hop connectivity* on the affinity graph — a soft,
> differentiable relaxation of the partition itself — instead of single edges
> (MALIS) or a non-differentiable greedy solver (MWS).

---

## Slide 1 — The gap: local edges, a global metric

The affinity head is trained per-edge (BCE / Dice / focal against affinity
targets), but the model is graded on **VI / adapted-Rand**, which measure
split/merge errors — i.e. the **connected-component** structure. Per-edge
accuracy is only a *proxy* for a topological objective.

| Method | Operates on | Differentiable? | Key weakness |
|---|---|:---:|---|
| Mutex Watershed | greedy edge sort + union-find | no | one spurious high-priority edge → catastrophic merge |
| MALIS | single maximin edge per pixel pair | yes | sparse gradient (1 edge/pair); O(N) maximin pass |
| **k-hop co-association** | **multi-hop paths (k-step diffusion)** | **yes** | oversmoothing if k too large (tunable) |

Goal: an objective that is transitivity-aware, differentiable, and cheap enough
for petascale EM.

---

## Slide 2 — Core idea: relax the *partition*, not the algorithm

Two voxels belong to the same instance **iff a high-affinity path connects
them**. Approximate that reachability with **k steps of sparse diffusion** over
the predicted affinity graph → a soft, differentiable `P(same instance)` for any
pair.

Why per-edge fails: if one weak edge bridges object A and object B, MWS may
merge them outright and MALIS puts gradient on that single edge. k-hop needs an
entire *path* of high affinity, so a lone spurious edge barely raises
connectivity — robust by construction.

```
   object A                         object B
   ●───●                             ●───●
    \ / \        one spurious         / \ /
     ●   ●- - - - - - - - - - - - - -●   ●
              (dashed weak edge)
  strong within-object edges         strong within-object edges
```

---

## Slide 3 — Mechanism

```python
A = sparse_affinity_graph(aff)   # sparse [N, N], edge weight = predicted affinity
x = node_features                # or one-hot-ish seeds
for _ in range(k):               # k sparse SpMVs
    x = normalize(A @ x)         # identity diffusion
s = cosine(x[i], x[j])           # soft P(same instance), in [0, 1]
loss_conn = BCE(s, same_gt[i, j])
loss = loss_aff + w_conn * loss_conn
```

1. **Build** the sparse affinity adjacency from the 30-offset stencil.
2. **Diffuse** identity `k` hops (normalized sparse aggregation).
3. **Score** sampled pairs at mixed z/y/x distances → soft co-association `s(i,j)`.
4. **Supervise** with BCE vs `[same GT instance]`; gradient flows into the
   affinity logits along *all* paths.

---

## Slide 4 — Why k-hop specifically (the design choice)

- **Transitivity-aware** — merges are transitive closures; k-hop reachability
  models exactly the property MWS computes, which per-edge losses cannot.
- **Path-robust** — a merge requires a whole high-affinity path, so a single
  mis-predicted edge can't bridge two objects. Directly hardens against MWS's
  catastrophic-merge failure mode.
- **Dense gradient** — diffusion spreads gradient over every path (vs MALIS's
  single edge per pair) → richer signal on the edges that (de)couple objects.
- **Cheap & local** — `k` sparse SpMVs over a k-hop neighborhood on sampled
  pairs; GPU-native, patch-local, no combinatorial solver in the training loop.

> One line: k-hop is the smallest change that turns a per-edge proxy into a
> transitivity-aware, path-robust, differentiable objective — without a solver
> in the loop.

---

## Slide 5 — Scales to petascale

| | |
|---|---|
| **O(k·E)** | cost — k sparse SpMVs, E = N × #offsets |
| **k ≈ 4–8** | hops (small; tune vs oversmoothing) |
| **sampled** | pairs per step — never the N×N matrix |

- **No dense co-association** — the full N×N matrix is never formed; only sampled
  pairs within the k-hop receptive field are scored.
- **Patch-local** — everything lives inside the training crop; no global graph,
  so it composes with block-wise EM pipelines.
- **GPU-native** — sparse matvecs on the affinity stencil are cheap next to a
  DiT forward.

---

## Slide 6 — Risks & mitigations

| Risk | Mitigation |
|---|---|
| Oversmoothing — large k blurs distinct objects | keep k small (4–8); sharpen affinities (temperature); anneal k |
| Leakage — soft connectivity ≠ exact partition | auxiliary term only; hard MWS still runs at inference on cleaner affinities |
| Not end-to-end through inference | by design — trains topologically-clean affinities; MWS stays the partitioner |

**Honest framing.** A design proposal, not a validated recipe. It touches known
primitives (graph diffusion, co-association); the bet is the scalable
combination for an affinity EM head. Not guaranteed to beat a well-tuned
MWS + MALIS — validate with an ablation first.

---

## Slide 7 — Integration into nanoCosmos

- Add a `weight_conn` term to `AffinityFGLoss`, gated like the existing
  `weight_aff` / `weight_sem` fields — off by default.
- Reuse the configured `offsets` as the diffusion stencil; sample pairs per crop
  at mixed z/y/x distances.
- Keep `MutexWatershed` at inference — it now agglomerates affinities trained to
  be transitively consistent.

**Ablation plan (one labeled volume).** Compare `affinity-only` vs
`affinity + w_conn·conn` on held-out crops; metric = VI (split / merge) after
MWS. Success = lower VI at equal affinity BCE, especially fewer *merge* errors
(the path-robustness claim).
