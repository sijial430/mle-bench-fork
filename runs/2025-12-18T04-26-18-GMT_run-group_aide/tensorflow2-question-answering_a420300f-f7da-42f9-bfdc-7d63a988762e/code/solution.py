import json
import os
import re
from collections import Counter
import pandas as pd

# Parameters
INPUT_DIR = "./input"
TRAIN_FILE = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
TEST_FILE = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# How many train examples to use for quick validation (to keep runtime reasonable)
N_TRAIN_SAMPLE = 10000  # adjust if needed

os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Utilities
_norm_re = re.compile(r"\w+", re.UNICODE)


def normalize_token(tok):
    m = _norm_re.findall(tok)
    if not m:
        return ""
    return "".join(m).lower()


def tokenize_whitespace(text):
    # Returns list of original tokens (as in dataset) and normalized tokens parallelly
    toks = text.split()
    norm = [normalize_token(t) for t in toks]
    return toks, norm


def predict_for_example(question_text, doc_tokens, doc_norm_tokens, long_candidates):
    # Build question token set
    q_norm_tokens = [normalize_token(t) for t in question_text.split()]
    q_set = set([t for t in q_norm_tokens if t])
    # If question token set empty, return blanks
    if not q_set:
        return "", ""
    best_score = 0
    best_cand = None
    # Evaluate each long candidate by count of token overlap (sum of occurrences)
    for cand in long_candidates:
        s = cand.get("start_token", -1)
        e = cand.get("end_token", -1)
        if s is None or e is None or s < 0 or e <= s or s >= len(doc_norm_tokens):
            continue
        e = min(e, len(doc_norm_tokens))
        cand_norm = doc_norm_tokens[s:e]
        if not cand_norm:
            continue
        score = sum(1 for t in cand_norm if t in q_set)
        if score > best_score:
            best_score = score
            best_cand = (s, e)
    if best_cand is None or best_score == 0:
        return "", ""
    long_s, long_e = best_cand
    pred_long = f"{long_s}:{long_e}"
    # Short answer: sliding window within long span, lengths 1..5
    best_s = None
    best_se = None
    best_short_score = 0
    max_window = 5
    L = long_e - long_s
    for w in range(1, max_window + 1):
        if w > L:
            break
        for start in range(long_s, long_e - w + 1):
            end = start + w
            window_norm = doc_norm_tokens[start:end]
            if not window_norm:
                continue
            score = sum(1 for t in window_norm if t in q_set)
            if score > best_short_score:
                best_short_score = score
                best_s = start
                best_se = end
    if best_short_score == 0:
        pred_short = ""
    else:
        pred_short = f"{best_s}:{best_se}"
    return pred_long, pred_short


# Functions to extract gold labels from training annotations
def extract_gold_labels_from_annotations(annotations):
    long_set = set()
    short_set = set()
    for ann in annotations:
        # Long answers
        la = ann.get("long_answer", {})
        if la:
            s = la.get("start_token", -1)
            e = la.get("end_token", -1)
            if s is not None and e is not None and s >= 0 and e > s:
                long_set.add(f"{s}:{e}")
        # Short answers
        sas = ann.get("short_answers", [])
        for sa in sas:
            ss = sa.get("start_token", -1)
            se = sa.get("end_token", -1)
            if ss is not None and se is not None and ss >= 0 and se > ss:
                short_set.add(f"{ss}:{se}")
        # Yes/No answers (if present)
        yesno = ann.get("yes_no_answer", None)
        if yesno:
            # Some datasets use 'YES','NO' or 'yes','no'
            yesno_up = str(yesno).upper()
            if yesno_up in ("YES", "NO"):
                short_set.add(yesno_up)
    return long_set, short_set


# Read a sample of training data and evaluate heuristic
def evaluate_on_train_sample(n_samples=N_TRAIN_SAMPLE):
    tp = fp = fn = 0
    count = 0
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if count >= n_samples:
                break
            obj = json.loads(line)
            question = obj.get("question_text", "")
            doc_text = obj.get("document_text", "")
            doc_tokens, doc_norm = tokenize_whitespace(doc_text)
            candidates = obj.get("long_answer_candidates", [])
            preds_long, preds_short = predict_for_example(
                question, doc_tokens, doc_norm, candidates
            )
            annotations = obj.get("annotations", [])
            gold_long_set, gold_short_set = extract_gold_labels_from_annotations(
                annotations
            )
            # Long evaluation
            if preds_long:
                if preds_long in gold_long_set:
                    tp += 1
                else:
                    fp += 1
            else:
                # predicted blank
                if gold_long_set:
                    fn += 1
            # Short evaluation
            if preds_short:
                if preds_short in gold_short_set:
                    tp += 1
                else:
                    fp += 1
            else:
                if gold_short_set:
                    fn += 1
            count += 1
            if count % 2000 == 0:
                print(f"Processed {count} train examples...")
    # compute micro F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "n": count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# Run evaluation on a sample of training data
print("Evaluating heuristic on a sample of training data...")
eval_stats = evaluate_on_train_sample()
print(
    f"Validation (sampled) micro F1: {eval_stats['f1']:.6f} (Precision {eval_stats['precision']:.4f}, Recall {eval_stats['recall']:.4f})"
)

# Now run predictions over test set and create submission.csv
print("Generating predictions for test set and writing submission file...")
# We'll stream test file and build rows
rows = []
test_count = 0
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        ex_id = str(obj.get("example_id"))
        question = obj.get("question_text", "")
        doc_text = obj.get("document_text", "")
        doc_tokens, doc_norm = tokenize_whitespace(doc_text)
        candidates = obj.get("long_answer_candidates", [])
        pred_long, pred_short = predict_for_example(
            question, doc_tokens, doc_norm, candidates
        )
        # Append two rows per example: _long and _short
        rows.append({"example_id": f"{ex_id}_long", "PredictionString": pred_long})
        rows.append({"example_id": f"{ex_id}_short", "PredictionString": pred_short})
        test_count += 1
        if test_count % 5000 == 0:
            print(f"Processed {test_count} test examples...")
print(f"Total test examples processed: {test_count}")

# Ensure order matches sample_submission if present; otherwise write rows as-is
if os.path.exists(SAMPLE_SUB_PATH):
    sample_df = pd.read_csv(SAMPLE_SUB_PATH)
    # Build dict for quick lookup
    pred_dict = {r["example_id"]: r["PredictionString"] for r in rows}
    # Some sample_submission might have NaN; fill with predicted or empty string
    out_preds = []
    for idx, row in sample_df.iterrows():
        eid = str(row["example_id"])
        ps = pred_dict.get(eid, "")
        out_preds.append({"example_id": eid, "PredictionString": ps})
    out_df = pd.DataFrame(out_preds)
else:
    out_df = pd.DataFrame(rows)

# Save submission
out_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")

# Final print of evaluation metric
print(f"Final reported validation micro F1 on sampled train: {eval_stats['f1']:.6f}")
