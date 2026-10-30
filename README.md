# portal-tta

A controller that decides, per batch and under scarce labels, whether a
test-time intervention should be deployed at all.

Test-time adaptation can rescue a shifted batch and damage the next one:
updating normalisation statistics from a batch helps when the batch is a
fair sample of the shifted domain, and fails when the batch is mostly one
class, because the batch statistics then encode the class instead of the
domain. Average accuracy hides that failure. This project implements a
controller in the style of PORTAL (Zhang et al., arXiv:2610.37658): it
treats each available prediction rule as a candidate, freezes every
candidate's predictions before any label arrives, spends scarce labels
only on inputs where candidates disagree, stops querying when the
remaining disagreements cannot change the choice or are not worth a
label, and refreshes stale evidence on periodic probe batches. Acquired
labels also train a label-adapted candidate for later batches.

This is an independent implementation of the controller principles in the
paper's abstract and reported design. It uses no code from the paper's
repository, and it is evaluated on a synthetic prototype testbed, not on
image data. The paper's own numbers are reported separately below and
are not claimed as results of this code.

## What was measured

Six scenarios, 12 batches of 64 inputs each, 8 seeds (0-7). Every policy
sees the same streams within a seed, so differences are paired. All
numbers below are from `results/measured.json`, produced by
`scripts/run_experiment.py` on this testbed only. Accuracy is mean
per-batch accuracy; harm is accuracy lost relative to deploying no
intervention (`base`), floored at zero per batch and then averaged;
worst is the lowest single-batch accuracy; labels are labels acquired
per batch (out of 64).

| Scenario | Policy | Accuracy | Harm | Worst batch | Labels/batch |
|---|---|---|---|---|---|
| clean | portal | 0.905 | 0.001 | 0.855 | 5.354 |
| clean | always base | 0.901 | 0.000 | 0.842 | 0.000 |
| clean | always batch-norm | 0.891 | 0.016 | 0.824 | 0.000 |
| clean | naive (random queries) | 0.902 | 0.004 | 0.852 | 16.000 |
| clean | oracle (all labels) | 0.912 | 0.000 | 0.861 | 64.000 |
| corrupt, mixed arrivals | portal | 0.891 | 0.000 | 0.824 | 10.990 |
| corrupt, mixed arrivals | always base | 0.696 | 0.000 | 0.621 | 0.000 |
| corrupt, mixed arrivals | always batch-norm | 0.891 | 0.000 | 0.824 | 0.000 |
| corrupt, mixed arrivals | oracle | 0.891 | 0.000 | 0.824 | 64.000 |
| corrupt, class-block arrivals | portal | 0.724 | 0.008 | 0.299 | 14.760 |
| corrupt, class-block arrivals | always base | 0.694 | 0.000 | 0.252 | 0.000 |
| corrupt, class-block arrivals | always batch-norm | 0.247 | 0.466 | 0.168 | 0.000 |
| corrupt, class-block arrivals | oracle | 0.760 | 0.000 | 0.357 | 64.000 |
| clean, class-block arrivals | always batch-norm | 0.247 | 0.651 | 0.168 | 0.000 |
| mixed arrivals (mixed + block) | portal | 0.812 | 0.026 | 0.430 | 14.854 |
| mixed arrivals (mixed + block) | always base | 0.693 | 0.000 | 0.262 | 0.000 |
| mixed arrivals (mixed + block) | always batch-norm | 0.676 | 0.156 | 0.197 | 0.000 |
| mixed arrivals (mixed + block) | naive | 0.761 | 0.042 | 0.418 | 16.000 |
| mixed arrivals (mixed + block) | portal, no refresh | 0.765 | 0.066 | 0.338 | 13.000 |
| mixed arrivals (mixed + block) | oracle | 0.864 | 0.000 | 0.623 | 64.000 |
| drift (clean, then corrupt) | portal | 0.898 | 0.000 | 0.842 | 8.552 |
| drift (clean, then corrupt) | portal, no refresh | 0.897 | 0.000 | 0.834 | 5.990 |
| drift (clean, then corrupt) | always batch-norm | 0.891 | 0.008 | 0.824 | 0.000 |

Headline findings:

- **The intervention is a per-batch decision, not a global one.** Always
  deploying batch-norm adaptation scores 0.891 on corrupted mixed
  batches but 0.247 on corrupted class-block batches, with mean harm
  0.466 there. The controller scores 0.891 on the mixed case and
  0.724 on the block case, with mean harm 0.008. On mixed arrivals
  (alternating mixed and block batches) it scores 0.812 against 0.676
  for always batch-norm, mean harm 0.026 against 0.156, worst batch
  0.430 against 0.197.
- **Arrival order alone breaks the intervention, with no corruption
  at all.** On clean class-block batches, always batch-norm scores
  0.247 with harm 0.651. Its accuracy is identical with and without
  corruption, because per-batch standardisation is invariant to the
  affine corruption used here; only the arrival order matters.
- **Labels are spent where they can change the choice.** On clean
  streams the controller uses 5.354 labels per batch against the
  naive controller's 16.000, at accuracy 0.905 against 0.902. It
  queries only inputs where candidates disagree, and it queries
  nothing at all when every candidate agrees.
- **Refreshing stale evidence matters when the regime varies
  batch to batch.** On mixed arrivals, the controller with periodic
  probe batches scores 0.812; without refresh it scores 0.765. On a
  single clean-to-corrupt drift the difference is small (0.898
  against 0.897), because evidence from the clean half is not
  strongly misleading once the corruption arrives.
- **Portfolio expansion was neutral on this testbed.** Adding a weak
  candidate (batch statistics from the first 8 inputs only) moved the
  controller on mixed arrivals from 0.812 to 0.807 and the naive
  controller from 0.761 to 0.763. That does not reproduce the
  paper's portfolio-expansion result; see the paper numbers below.
  The honest reading is that this testbed's weak candidate is too
  weak to be chosen often, so the interaction the paper reports is
  not exercised here.

### The paper's numbers (not this project's)

Zhang et al. report, on PACS with class-block arrivals, that BN+flip
reaches 22.58% accuracy while their controller reaches 51.28%, and net
batch harm falls from 32.41 to 0.34 percentage points. Expanding the
candidate portfolio raises error by 14.32 points under a HILTTA-style
controller while lowering it by 1.24 points under theirs. Those are
image-classification results from the paper; the numbers in the table
above are simulation results from this project's testbed and must not
be compared directly with them.

## How the controller works

1. **Frozen candidates.** Each candidate's predictions for a batch are
   computed once, before any label for that batch. One acquired label
   therefore evaluates every candidate on the same input (paired
   comparisons), instead of each rule being judged on different inputs.
2. **Directed queries.** The controller queries inputs where a
   challenger disagrees with the current leader, preferring inputs
   where the most challengers disagree. Inputs where all candidates
   agree carry no information about which rule to deploy.
3. **Early stopping.** Querying stops when no challenger can still
   overtake the leader -- the leader's margin exceeds the number of
   remaining leader-challenger disagreements -- or when even winning
   every remaining disagreement would not improve the deployed
   accuracy margin by more than `label_cost` (default 0.01). The
   remaining-swing bound counts disagreements only, which is what
   makes it tight when most inputs are unanimous.
4. **Ties favour no intervention.** A candidate must beat `base` on
   acquired labels to be deployed; on a tie, `base` is chosen.
5. **Periodic refresh.** Every `probe_every` batches (default 4), the
   controller discards accumulated evidence and gathers fresh labels
   on a probe batch, so evidence from a past regime cannot keep
   pointing at the wrong rule after conditions change.
6. **Labels train the future.** Labels the controller actually bought
   update the label-adapted candidate's prototypes, after the batch
   they were bought for has been scored. No candidate is ever scored
   with labels bought for that same batch.

## Candidates

| Candidate | Rule |
|---|---|
| `base` | Nearest prototype, raw features. No intervention. |
| `batch_norm` | Per-batch recentring/rescaling, compared against source statistics. The intervention under test. |
| `shrink` | Batch statistics shrunk halfway toward source statistics. A conservative intervention. |
| `label_adapted` | Prototypes moved toward the means of acquired labels (EMA). Learns only from labels actually bought. |
| `batch_norm_first8` | Batch statistics from the first 8 inputs only. Weak candidate, expanded portfolio only. |

## Quickstart

```bash
pip install -r requirements.txt
python scripts/demo.py --n-batches 6 --seed 0
PYTHONPATH=src python -m portal_tta.cli demo --n-batches 6 --seed 0
```

Run the full experiment (6 scenarios x 9 policies x 8 seeds):

```bash
python scripts/run_experiment.py
```

This regenerates `results/experiment.json` (per-seed rows) and
`results/measured.json` (seed-averaged table and headline values).
Check the README figures against the regenerated results with:

```bash
python scripts/crossfoot.py
```

Use the controller in your own code by supplying candidates whose
predictions are frozen per batch and a label oracle:

```python
from portal_tta.controller import PortalConfig, PortalController, frozen_predictions
from portal_tta.candidates import make_candidates

candidates = make_candidates("core")
controller = PortalController(PortalConfig(max_labels=16, seed=0))
predictions = frozen_predictions(candidates, batch_x)
decision = controller.decide(batch_index, predictions, lambda idx: batch_y[idx])
deployed = predictions[decision.choice]
```

See `docs/ADAPTING_TO_REAL_MODELS.md` for replacing the synthetic
candidates with a real model and its interventions, and
`docs/RESEARCH.md` for the research trail: what was checked first,
what was rejected, and what this testbed does and does not show.

## Layout

```
src/portal_tta/
  data.py        synthetic streams: clean/corrupt x mixed/block arrivals, drift
  candidates.py  frozen prediction rules, including the learning candidate
  controller.py  the controller and the naive random-query controller
  policies.py    fixed, oracle, and controller policy wrappers
  benchmark.py   per-batch protocol: freeze, decide, score, then learn
  experiment.py  full grid and seed-averaged results
  metrics.py     accuracy, harm, worst-batch, regret
  cli.py         demo and benchmark commands
scripts/         demo, run_experiment, crossfoot
tests/           unit, controller, benchmark, and cross-foot tests
docs/            research trail and real-model adaptation guide
config/          experiment defaults
examples/        example controller configuration
```

## Limitations

- The testbed is synthetic: Gaussian blobs, a nearest-prototype
  classifier, and affine corruptions. It demonstrates the controller
  logic and the batch-statistics failure mode; it says nothing about
  any real model's accuracy under real distribution shift.
- Labels are assumed correct and equally costly. Noisy or adversarial
  labels, and labels whose cost varies by input, are out of scope.
- The label-adapted candidate learns from very few labels per batch;
  on this testbed it rarely wins, and most of the controller's gain
  comes from choosing between `base` and `batch_norm` per batch.
- Refresh is periodic and unconditional. Detecting drift from
  unlabelled inputs and probing only then would spend fewer labels;
  that is the natural next step.
