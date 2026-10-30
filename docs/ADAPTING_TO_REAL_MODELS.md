# Adapting the controller to a real model

The controller never sees features, weights, or gradients. It sees
only, per batch, one prediction vector per candidate, and it buys
labels through a callable. Adapting it to a real deployment means
replacing the candidate definitions and the label oracle; the
controller code does not change.

## What a candidate is in production

A candidate is any rule that can produce a prediction for every input
in a batch before any label for that batch is seen:

- `base`: the deployed model as-is, no test-time change.
- An adaptation rule: the same model with batch-norm statistics
  updated from the batch, with test-time augmentation, with a
  different decoding or retrieval setting, or with a lightweight
  adapter switched on.
- A learned rule: a model or head updated only from labels the
  controller has already bought, frozen for the current batch.

The freezing requirement is load-bearing. If a candidate's
predictions change after the first label of a batch arrives, the
paired-comparison property breaks and the evidence is no longer a
fair comparison. Compute all candidates' predictions first, then
query.

## Wiring

```python
candidates = {
    "base": lambda x: model.predict(x),
    "adapted": lambda x: model.predict_with_batch_stats(x),
}
predictions = {name: fn(batch_x) for name, fn in candidates.items()}
decision = controller.decide(batch_index, predictions, label_oracle)
deployed = predictions[decision.choice]
```

`label_oracle` should return true labels for exactly the indices it
is given, in order. In production that is usually a human review
queue, a delayed ground-truth feed, or a high-precision teacher
model -- in all cases, budget it: the controller's label cost is the
real cost of that oracle, and `label_cost` should be set from it, not
left at the simulation default.

## Settings to re-tune

- `max_labels`: the most labels the oracle can supply per batch.
  Set it from the review budget, not from accuracy alone.
- `label_cost`: the accuracy gain, in points, that one label must be
  worth. If labels are expensive or slow, raise it; the controller
  will stop earlier and lean on accumulated evidence.
- `probe_every` / `refresh_probe`: how often stale evidence is
  discarded. Shorten the interval when conditions change fast;
  lengthen it when they are stable, since every probe batch pays
  for fresh labels.
- `prefer`: the safe default. Keep it on the no-intervention rule
  unless there is positive evidence that an intervention is the
  safe default for the deployment.

## What to monitor in production

Report the same three numbers the simulation reports, per batch:
deployed accuracy on labelled inputs, harm relative to the
no-intervention rule, and the worst batch. Average accuracy alone
will hide exactly the failure this controller exists to prevent.
Also log, per batch, the chosen candidate, the number of labels
bought, and whether the batch was a probe batch: a controller that
suddenly buys its full budget every batch is telling you conditions
have changed faster than its evidence can follow.
