"""CIP-107 Phase C - reproducible OFFLINE classifier + rule-based baseline.
Usage: python3 phase_c_classifier.py dataset_train.jsonl dataset_test.jsonl [outdir]
No network, no OpenAI calls. Nothing here is a fine-tuned model run.
"""
import sys, os, re, json, ipaddress, random, platform, hashlib
import numpy as np, pandas as pd, sklearn, scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
random.seed(SEED); np.random.seed(SEED)
train_path, test_path = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else "phase_c_outputs"
os.makedirs(out, exist_ok=True)

def load(p):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return pd.DataFrame({"prompt": [r["prompt"] for r in rows],
                         "y": [1 if r["completion"].strip() == "positive" else 0 for r in rows]})
tr, te = load(train_path), load(test_path)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()

# ---------- 10. Leakage inspection (record, do not silently remove) ----------
def norm(s): return re.sub(r"\s+", " ", s).strip()       # ignore tab-indentation differences
tr["n"], te["n"] = tr.prompt.map(norm), te.prompt.map(norm)
exact_raw = te.prompt.isin(set(tr.prompt))
exact_norm = te.n.isin(set(tr.n))
lab_map = tr.groupby("n").y.agg(lambda s: set(s)).to_dict()
conflict = [(n, lab_map[n]) for n, y in zip(te.n, te.y) if n in lab_map and y not in lab_map[n]]
# near-identical: char-3..5 TF-IDF cosine >= 0.90, vectoriser fitted on TRAIN only
lv = TfidfVectorizer(analyzer="char", ngram_range=(3, 5)).fit(tr.n)
sim = cosine_similarity(lv.transform(te.n), lv.transform(tr.n))
te["max_sim"] = sim.max(1); te["nearest_train_label"] = tr.y.values[sim.argmax(1)]
near = te.max_sim >= 0.90
leak = {
 "train_rows": len(tr), "test_rows": len(te),
 "train_unique_prompts": int(tr.n.nunique()), "test_unique_prompts": int(te.n.nunique()),
 "test_rows_exact_duplicate_of_train_raw": int(exact_raw.sum()),
 "test_rows_exact_duplicate_of_train_ignoring_whitespace": int(exact_norm.sum()),
 "test_rows_near_identical_cosine>=0.90": int(near.sum()),
 "test_rows_with_no_near_match_(clean_subset)": int((~near).sum()),
 "duplicate_prompts_with_conflicting_labels_across_split": len(conflict),
 "pct_test_leaked_(near_or_exact)": round(100 * near.mean(), 1)}
te["leak_flag"] = np.where(exact_norm, "exact", np.where(near, "near", "none"))

# ---------- 9. TF-IDF + Logistic Regression (vectoriser fitted on TRAIN only) ----------
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, sublinear_tf=True)
Xtr = vec.fit_transform(tr.prompt)            # fit ONLY here
Xte = vec.transform(te.prompt)                # transform only
clf = LogisticRegression(C=10.0, class_weight="balanced", max_iter=1000, random_state=SEED, solver="liblinear")
clf.fit(Xtr, tr.y)
te["pred_tfidf_lr"] = clf.predict(Xte)
te["proba_pos"] = clf.predict_proba(Xte)[:, 1].round(4)

# ---------- 11. Transparent rule baseline (regex + ipaddress + context) ----------
SPLIT = re.compile(r"^(?P<text>.*);(?P<cand>[^;]*?)\s*->\s*$", re.S)
def rule_predict(prompt):
    """Return (label, reason). 1 = positive (usable IPv4 address), 0 = negative."""
    m = SPLIT.match(prompt)
    if not m: return 0, "R0 no 'text;candidate ->' structure"
    text, cand = m.group("text"), m.group("cand").strip()
    try: ip = ipaddress.IPv4Address(cand)
    except ipaddress.AddressValueError: return 0, "R1 candidate not valid IPv4 (ipaddress)"
    if ip.is_unspecified: return 0, "R2 0.0.0.0 placeholder"
    if int(cand.split(".")[0]) == 0: return 0, "R3 first octet 0 ('this network', not a host)"
    pos = text.find(cand)
    if pos < 0: return 0, "R4 candidate not found in text"
    before, after = text[:pos], text[pos + len(cand):]
    if re.search(r"\d\.$", before) or re.match(r"\.\d", after):
        return 0, "R5 fragment of a longer dotted number (OID/version/hostname)"
    if re.search(r"(?i)(/|release|releases|rc-|crop\.)[^<]*$", before) and re.search(r"https?://", text) and ("/" in after or "." in after.split("<")[0]):
        return 0, "R6 inside URL path / file name"
    return 1, "R7 valid, standalone IPv4"
res = [rule_predict(p) for p in te.prompt]
te["pred_rule"] = [r[0] for r in res]; te["rule_reason"] = [r[1] for r in res]

# ---------- metrics ----------
def block(name, y, p):
    cm = confusion_matrix(y, p, labels=[0, 1])
    d = {"name": name, "n": int(len(y)), "confusion_matrix_[[TN,FP],[FN,TP]]": cm.tolist(),
         "accuracy": round(accuracy_score(y, p), 4),
         "precision_pos": round(precision_score(y, p, zero_division=0), 4),
         "recall_pos": round(recall_score(y, p, zero_division=0), 4),
         "f1_pos": round(f1_score(y, p, zero_division=0), 4)}
    rep = classification_report(y, p, labels=[0, 1], target_names=["negative", "positive"], zero_division=0)
    return d, rep
metrics, reports = [], []
clean = te.leak_flag == "none"
for nm, mask in [("PRIMARY: full supplied holdout", np.ones(len(te), bool)),
                 ("SECONDARY: holdout minus exact/near-duplicates of train", clean.values)]:
    if mask.sum() == 0: continue
    for model, col in [("TF-IDF+LogReg", "pred_tfidf_lr"), ("Rule baseline", "pred_rule")]:
        d, rep = block(f"{model} | {nm}", te.y[mask], te[col][mask]); metrics.append(d)
        reports.append(f"===== {d['name']} (n={d['n']}) =====\n{rep}\nConfusion matrix [[TN FP][FN TP]]: {d['confusion_matrix_[[TN,FP],[FN,TP]]']}\n")
# majority-class reference so accuracy is not over-read
maj = int(tr.y.mode()[0]); d, rep = block("Majority-class reference | full holdout", te.y, np.full(len(te), maj)); metrics.append(d)

te.drop(columns=["n"]).to_csv(f"{out}/test_predictions.csv", index=False)
json.dump({"leakage": leak, "metrics": metrics}, open(f"{out}/metrics.json", "w"), indent=2)
open(f"{out}/classification_reports.txt", "w").write("\n".join(reports))
open(f"{out}/environment.txt", "w").write("\n".join([
 f"python {platform.python_version()}", f"scikit-learn {sklearn.__version__}", f"numpy {np.__version__}",
 f"pandas {pd.__version__}", f"scipy {scipy.__version__}", f"random_seed {SEED}",
 "vectoriser: TfidfVectorizer(analyzer=char_wb, ngram_range=(2,5), sublinear_tf=True) fitted on train only",
 "model: LogisticRegression(C=10, class_weight=balanced, solver=liblinear, random_state=42)",
 f"train_sha256 {sha(train_path)}", f"test_sha256 {sha(test_path)}",
 "fine_tuned_model_run: NONE (offline classical models only)"]))
print(json.dumps(leak, indent=1)); print("\n".join(reports))
print(te[te.pred_rule != te.y][["prompt", "y", "pred_rule", "rule_reason"]].to_string(max_colwidth=70))
print(te[te.pred_tfidf_lr != te.y][["prompt", "y", "pred_tfidf_lr", "leak_flag"]].to_string(max_colwidth=70))
