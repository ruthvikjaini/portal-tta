# Research trail — 30 October 2026

## Trending check first

Checked before any paper reading or code, as required. GitHub
trending on 29-30 October 2026 was again dominated by agent skill
packs and prompt collections (superpowers, ECC, hallmark, impeccable,
design skills), plus a small number of large agent frameworks and
inference engines. The candidates with code and a measurable weakness
were judged as follows:

- **Skill/prompt packs** (obra/superpowers, affaan-m/ECC,
  Nutlope/hallmark, pbakaus/impeccable, and similar): no runnable
  code with a benchmarkable weakness; a re-skin would add nothing
  measurable. Rejected.
- **deepseek-ai/DeepInfer** (23.0k stars, Apache-2.0, week of
  19-25 Oct): a production inference engine for one model, NVIDIA
  SM100 / Huawei Ascend only, CUDA 13 / CANN 9.2. No CPU-runnable
  benchmark is possible in this environment, and the measurable
  weakness would be in GPU kernels. Rejected on CPU-runnability.
- **headroomlabs-ai/headroom** (76k stars): context compression for
  agent transcripts. Close in spirit to the Day 5 KV-cache project
  (compression under a budget) and its claims are its own benchmarks
  on proprietary agent traces that cannot be reproduced here.
  Rejected as too close to Day 5.
- **HKUDS/DeepTutor** (49k stars): a full tutoring platform, not a
  component with a single measurable weakness. Rejected.
- **morluto/rea, Gathin23/cleanyourtv, debpalash/VoiceStudio**:
  reverse-engineering agents, Android TV debloat, and a local voice
  studio. Outside the Gen AI / ML / RL brief, or (for the voice
  studio) a GPU-heavy product build. Rejected.

Conclusion, same as the previous night: no trending repo offered a
genuine, measurable, CPU-runnable weakness worth optimising. The
build therefore falls back to a paper-based original project, per the
standing rule. No trending repo's code is used, and no trending repo
is presented as the basis of this project.

## Papers considered

### Chosen: PORTAL — "When Should We Intervene at Test Time?"

- Zhang, Wen, Li, Chen. arXiv:2610.37658v1, submitted 27 October
  2026. Abstract verified on the arXiv abstract page on 30 October
  2026: https://arxiv.org/abs/2610.37658v1
- Core claim: test-time adaptation (e.g. updating normalisation
  statistics) can rescue one batch and damage the next; average
  accuracy hides the damage. The controller chooses when to deploy
  an intervention, which inputs to label, and when to stop querying.
  Each prediction rule is a candidate with predictions fixed before
  feedback, so every acquired label evaluates all candidates on the
  same input. Queries go to candidates that can still overtake the
  leader; querying stops when remaining labels cannot change the
  choice or the possible gain is not worth another query. Periodic
  label checks refresh stale evidence. Acquired labels also train
  stronger candidates for future batches.
- Reported results (paper's, on PACS): under class-block arrivals,
  BN+flip 22.58% vs controller 51.28%; net batch harm 32.41 to 0.34
  percentage points. Expanding the portfolio raises error 14.32
  points under a HILTTA-style controller and lowers it 1.24 points
  under theirs. These numbers are reported in the README as the
  paper's only; they are not reproduced here.
- Why it fits: the controller is an algorithm over frozen
  predictions and labels, so it can be implemented faithfully and
  evaluated exactly on a synthetic testbed without a GPU, a real
  model, or labelled image data. The failure mode it targets --
  batch statistics under grouped arrivals -- is reproducible with
  elementary statistics, which was verified in a prototype before
  building (batch-norm accuracy 0.931 on corrupted mixed batches,
  0.333 on class-block batches, in the prototype geometry).
- What was taken: the controller principles listed above. What was
  not taken: the paper's code (an anonymous repository link in the
  paper was not used), its image models, or its datasets. The
  candidates here are prototype-classifier analogues of
  normalisation-stat adaptation, not the paper's BN/flip rules.

### Considered and rejected: MPO — "Multiscale Policy Optimization for Long-Horizon LLM Agents"

- Ren et al. arXiv:2610.35013, submitted 26 October 2026. Abstract
  verified on the arXiv abstract page on 30 October 2026:
  https://arxiv.org/abs/2610.35013
- Idea: critic values vary mainly across turns, so credit assignment
  should be anchored at the turn scale (multiscale advantage
  estimation plus turn-level importance sampling), over token-level
  GRPO-style baselines.
- Rejected for tonight: it extends the Day 1 GRPO project and sits
  next to the Day 4 fault-harness work, so it would be the third
  project in the same agent-RL cluster; the abstract gives no
  per-task numbers or baseline names to anchor an independent
  replication; and a faithful version needs real LLM rollouts, which
  this CPU-only setup cannot run. The turn-scale credit idea remains
  a strong candidate for a later night with a GPU-backed rollout
  harness. The related TreeOPSD paper (arXiv:2610.23181, turn-as-tree
  -node credit backup) was noted in the same pass and set aside for
  the same reason.

### Considered and rejected: outcome-RL finishing-vs-reach audit

- Yang, "Outcome RL Rarely Expands Capability: 82% of Pass@1 Gains
  Are Just Finishing", arXiv:2610.34894, 29 October 2026 (via
  secondary coverage; abstract not verified on arXiv in this pass).
- The decomposition (finishing vs reach across 20 released RL
  models) is a measurement result about existing models, not a
  buildable method; an audit tool built on simulated policies would
  mostly re-state the paper's definitions. Rejected as a weaker
  build than a working controller.

### Held for a later night: Barron et al., "A Fix for Softmax and Cross-Entropy"

- Barron, Gnanapragasam, Tschannen (Google DeepMind),
  arXiv:2610.31642, 22 October 2026 (v2 26 October). Two numerical
  bugs in standard softmax/cross-entropy: softmax collapse past a
  precision cliff (margin about 5.5 in bf16, 16.6 in fp32) and
  softmax cancellation in distillation.
- Held, not rejected: a faithful reproduction needs bf16 training
  runs (e.g. the paper's modular-addition grokking benchmark) with
  controlled rounding, which deserves a dedicated night with a
  low-precision training harness rather than being rushed alongside
  the controller build.

## Testbed assumptions (explicit)

1. Classes are Gaussian blobs (4 classes, 8 dimensions, sigma 0.7)
   and the classifier is nearest-prototype. Real classifiers are
   not this, but the batch-statistics failure depends only on what
   a batch's mean and standard deviation estimate, which this
   reproduces exactly.
2. A domain is an affine corruption: per-feature gain drawn as
   exp(N(0, 0.9)) and shift drawn as N(0, 1), fixed per stream.
   Per-batch standardisation is invariant to affine corruptions,
   which is why batch-norm accuracy is identical on clean and
   corrupted class-block batches in the results.
3. Class-block arrival means one class per batch, rotating. This is
   the strong form of grouped arrival; real grouped arrivals are
   usually partial (one class dominant, not exclusive), so the
   measured harm is an upper-bound-style case, not a typical case.
4. Labels are correct, equally costly, and available on demand for
   queried inputs. The controller's stopping rule prices a label at
   `label_cost` accuracy points (default 0.01); changing that price
   changes how many labels are bought, and the grid result should be
   re-run if it is changed.
5. All reported numbers are simulation numbers on this testbed.
   Nothing here is a claim about PACS, ImageNet, or any deployed
   model.

## Sources

- Zhang et al., arXiv:2610.37658v1 (27 Oct 2026) — controller design.
- Ren et al., arXiv:2610.35013 (26 Oct 2026) — considered, rejected.
- Barron et al., arXiv:2610.31642 (22 Oct 2026) — held for later.
- GitHub trending and weekly-ranking pages consulted 30 Oct 2026
  (findarepo, gittrend.io, whatstrending.ai, dev.to trending
  round-ups) — trending check; no repo used.
