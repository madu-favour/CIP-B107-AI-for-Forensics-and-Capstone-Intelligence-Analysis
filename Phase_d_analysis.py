"""CIP-107 Phase D - analyse HISTORICAL fine-tuning logs (read-only; nothing is trained or sent to any API).
Usage: python3 phase_d_analysis.py code/result.csv code/fine_tuning_steps.txt [outdir]
"""
import sys, os, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res_path, steps_path = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else "phase_d_outputs"
os.makedirs(out, exist_ok=True)
r, s = pd.read_csv(res_path), pd.read_csv(steps_path)

# ---- 13. what each file contains, and are they the same run? ----
m = s.merge(r, on="step", suffixes=("_s", "_r"))
summary = {
 "fine_tuning_steps.txt": {"rows": len(s), "columns": list(s.columns), "last_step": int(s.step.max())},
 "result.csv": {"rows": len(r), "columns": list(r.columns), "last_step": int(r.step.max()),
                "validation_rows_present": int(r.validation_loss.notnull().sum()),
                "classification_rows_present": int(r["classification/accuracy"].notnull().sum())},
 "same_run_check": {"matching_steps": len(m),
                    "max_abs_training_loss_difference": round(float((m.training_loss_s - m.training_loss_r).abs().max()), 4),
                    "step1_loss_steps_txt": round(float(s.training_loss.iloc[0]), 4),
                    "step1_loss_result_csv": round(float(r.training_loss.iloc[0]), 4)}}
for name, d in [("fine_tuning_steps.txt", s), ("result.csv", r)]:
    summary[name].update({"mean_loss_first50": round(float(d.training_loss.iloc[:50].mean()), 4),
                          "mean_loss_last50": round(float(d.training_loss.iloc[-50:].mean()), 4),
                          "min_loss": round(float(d.training_loss.min()), 5),
                          "steps_with_train_seq_acc_0": int((d.training_sequence_accuracy == 0).sum()),
                          "mean_train_seq_acc": round(float(d.training_sequence_accuracy.mean()), 4)})
v = r[r.validation_loss.notnull()]
summary["validation"] = {"points": len(v), "min_val_loss": round(float(v.validation_loss.min()), 5),
                         "min_val_loss_step": int(v.step[v.validation_loss.idxmin()]),
                         "final_val_loss": round(float(v.validation_loss.iloc[-1]), 5),
                         "steps_with_val_seq_acc_below_1": v[v.validation_sequence_accuracy < 1].step.astype(int).tolist()}

# ---- 15. classification metrics: all non-null rows, then the final one ----
cols = ["step", "classification/accuracy", "classification/precision", "classification/recall",
        "classification/f1.0", "classification/auroc", "classification/auprc"]
c = r[r["classification/accuracy"].notnull()][cols]
c.to_csv(f"{out}/classification_rows.csv", index=False)
summary["final_non_null_classification_metrics"] = {k.split("/")[-1]: round(float(c.iloc[-1][k]), 6) for k in cols[1:]} | {"step": int(c.iloc[-1].step)}
summary["missing_values"] = {col: int(r[col].isnull().sum()) for col in r.columns if r[col].isnull().any()}
json.dump(summary, open(f"{out}/phase_d_summary.json", "w"), indent=2)

# ---- 14. plots ----
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
a = ax[0, 0]
a.plot(r.step, r.training_loss, color="#ccc", lw=.7, label="training loss (per step)")
a.plot(r.step, r.training_loss.rolling(25, min_periods=1).mean(), color="#c0392b", lw=2, label="training loss, 25-step average")
a.plot(v.step, v.validation_loss, "o-", color="#1f4e79", ms=3, lw=1.2, label=f"validation loss ({len(v)} recorded points)")
a.set_yscale("log"); a.set_xlabel("step"); a.set_ylabel("loss (log scale)"); a.set_title("A. result.csv: loss vs step"); a.legend(fontsize=7)
a = ax[0, 1]
a.plot(r.step, r.training_sequence_accuracy.rolling(25, min_periods=1).mean(), color="#c0392b", lw=2, label="training sequence accuracy (25-step average)")
a.plot(v.step, v.validation_sequence_accuracy, "o-", color="#1f4e79", ms=3, lw=1.2, label="validation sequence accuracy")
a.set_ylim(-.05, 1.05); a.set_xlabel("step"); a.set_title("B. result.csv: sequence accuracy vs step"); a.legend(fontsize=7, loc="lower right")
a = ax[1, 0]
for col, mk in zip(cols[1:], "os^Dx+"):
    a.plot(c.step, c[col], marker=mk, lw=1, label=col.split("/")[-1])
a.set_xlim(0, r.step.max() + 30); a.set_ylim(.92, 1.005); a.set_xlabel("step")
a.set_title(f"C. result.csv: classification metrics\n(only {len(c)} recorded points)"); a.legend(fontsize=7, ncol=2)
a = ax[1, 1]
a.plot(s.step, s.training_loss.rolling(25, min_periods=1).mean(), color="#7d3c98", lw=2, label=f"fine_tuning_steps.txt ({len(s)} steps)")
a.plot(r.step, r.training_loss.rolling(25, min_periods=1).mean(), color="#c0392b", lw=2, ls="--", label=f"result.csv ({len(r)} steps)")
a.set_yscale("log"); a.set_xlabel("step"); a.set_title("D. The two logs are different runs\n(training loss, 25-step average)"); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{out}/phase_d_plots.png", dpi=130)

print(json.dumps(summary, indent=1)); print("\nAll non-null classification rows:\n", c.to_string(index=False))
print(f"\nSaved to {out}/: phase_d_plots.png, classification_rows.csv, phase_d_summary.json")
