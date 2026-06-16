# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
[blank — write out the accuracy formula in plain English.
 What counts as "correct"? What do you divide by?]
```
accuracy = number of correct predictions / total predictions.
A prediction is correct when predictions[i] exactly equals ground_truth[i]
(same index, exact string match). Divide by the number of predictions.
---

**Step-by-step logic:**

```
[blank — describe the steps your code will take.
 1. ...
 2. ...
 3. ...]
```
1. If predictions is empty, return 0.0.
2. Zip predictions and ground_truth together.
3. Count the pairs where the two are exactly equal.
4. Divide that count by len(predictions) and return the float.

---

**Edge case — what if both lists are empty?**

```
[what should the function return? Why?]
```
Return 0.0. There's nothing to score, and dividing by zero would crash.
0.0 is the safe "no correct predictions out of nothing" value.
---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

[ what does compute_accuracy() return for these inputs? Show your work.]
```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

idx0  interview == interview  ✓
idx1  solo      == solo       ✓
idx2  panel     != solo       ✗
idx3  interview != narrative  ✗

2 correct / 4 total = 0.5  → returns 0.5
---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
[blank — be precise. When does an episode count as correctly classified
 for the "interview" class, for example?]
```
For class C, an episode is counted as correct when its ground-truth label is C AND the prediction for that episode is also C. Grouping is by ground-truth label,
not predicted label.
---

**What does "total" mean for a given class?**

```
[blank — is "total" the total number of predictions, or something more specific?]
```
The number of episodes whose GROUND-TRUTH label is this class, not the number of times the model predicted it. The key decision is grouping by ground truth makes the metric recall per class. Grouping by predicted label would
compute something different (precision-flavored) and is the classic bug that
makes a failing class look fine.
---

**Step-by-step logic:**

```
[blank — describe the steps your code will take.
 1. Initialize ...
 2. Loop over ...
 3. For each pair (predicted, truth) ...
 4. After the loop ...
 5. Return ...]
```
1. For each label in VALID_LABELS:
2.   total   = count of ground_truth entries equal to that label
3.   correct = count of positions where BOTH prediction and ground_truth
               equal that label (zip the two lists)
4.   accuracy = correct / total, or 0.0 when total is 0
5.   store {"correct", "total", "accuracy"} under that label
6. Return the dict.

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
[blank — what should accuracy be set to? Why?
 Hint: look at the docstring in evaluate.py.]
```
Set that class's accuracy to 0.0. It avoids dividing by zero, and the report's
bar-chart math expects a float (not None). It means "no episodes of this class
to score," which the report still displays cleanly.
---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

[blank — fill in the per-class results table below]

label       correct  total  accuracy
----------  -------  -----  --------
interview      1       1      1.00
solo           1       2      0.50
panel          1       1      1.00
narrative      0       1      0.00
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
Why is per-class accuracy a more informative metric than overall accuracy alone?

Overall accuracy can stay high while one class fails completely, because a
dominant class carries the number. A classifier that's 100% on the largest
class and 0% on a small one can still post a respectable overall score —
per-class is what exposes that the model never actually learned that class.




If panel episodes consistently get misclassified as interview, what does
that tell you about your training labels or your prompt?

The panel-vs-interview boundary is underspecified in the prompt: the
few-shot examples don't clearly distinguish multi-guest roundtables (equal
airtime, no host-guest dynamic) from two-person host-guest conversations,
so the model defaults to the more familiar label. The fix is in the
examples, not the model — add clearer, more prototypical panel cases.




2. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
How might the evaluation results change if you had labeled 100 training episodes?
What if you had 200 test episodes?

3. More training examples give a richer signal and most likely help the weakest
boundaries (panel/interview) first, with diminishing returns plus higher
prompt cost and latency. More test episodes make the accuracy estimate more
reliable.
With 5 per class a single error swings a class by 20 points,
50 per class each error moves it by 2, so small differences become trustworthy.