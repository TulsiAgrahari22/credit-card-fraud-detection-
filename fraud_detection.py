# ==============================================================================
# FRAUD DETECTION: ROBUST TWO-STAGE CASCADE (Logistic Regression -> XGBoost)
# Single cell: data loading -> split -> auto threshold tuning -> final report
# ==============================================================================
import os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ------------------------------------------------------------------------------
# STEP 0: LOAD DATA (path khud dhoondh leta hai)
# ------------------------------------------------------------------------------
CANDIDATES = [
    "/kaggle/input/datasets/organizations/mlg-ulb/creditcardfraud/creditcard.csv",
    "/kaggle/input/creditcardfraud/creditcard.csv",
]
DATA_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
if DATA_PATH is None:                       # dono path na milein to /kaggle/input mein search
    for root, _, files in os.walk("/kaggle/input"):
        if "creditcard.csv" in files:
            DATA_PATH = os.path.join(root, "creditcard.csv")
            break
assert DATA_PATH is not None, "creditcard.csv nahi mili: Kaggle mein Add Input se dataset jodo"

df = pd.read_csv(DATA_PATH)
print(f"[OK] Loaded {DATA_PATH} | shape = {df.shape}")



import matplotlib.pyplot as plt

print("Shape:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
print("\nClass counts:\n", df["Class"].value_counts())
print(f"Fraud rate: {df['Class'].mean() * 100:.3f}%")

print("\nAmount by class:")
print(df.groupby("Class")["Amount"].describe().round(2))

df["Hour"] = (df["Time"] // 3600) % 24
fraud_by_hour = df.groupby("Hour")["Class"].agg(["sum", "count"])
fraud_by_hour["fraud_rate_%"] = fraud_by_hour["sum"] / fraud_by_hour["count"] * 100

corr = df.drop(columns=["Hour"]).corr()["Class"].drop("Class").sort_values()
print("\nMost negative correlation with fraud:\n", corr.head(5).round(3))
print("\nMost positive correlation with fraud:\n", corr.tail(5).round(3))

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
df["Class"].value_counts().plot(kind="bar", ax=ax[0], title="Class balance")
np.log1p(df["Amount"]).hist(bins=50, ax=ax[1]); ax[1].set_title("log(1 + Amount)")
fraud_by_hour["fraud_rate_%"].plot(kind="bar", ax=ax[2], title="Fraud rate % by hour")
plt.tight_layout(); plt.show()

df = df.drop(columns=["Hour"])   # remove helper column so it does not enter the model

# ------------------------------------------------------------------------------
# STEP 1: TRAIN / TEST SPLIT (test set sirf final report ke liye)
# ------------------------------------------------------------------------------
X = df.drop(columns=["Class"])
y = df["Class"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Train = {len(X_train)} (frauds {int(y_train.sum())}) | "
      f"Test = {len(X_test)} (frauds {int(y_test.sum())})")


def _take(A, idx):
    return A.iloc[idx] if hasattr(A, "iloc") else A[idx]


# ------------------------------------------------------------------------------
# STEP 2: CASCADE CLASS (thresholds kai random 80/20 splits par tune hote hain)
# ------------------------------------------------------------------------------
class RobustCascade:
    """Stage 1 (LR): p < T_low -> genuine (final) | p > T_high -> fraud (final)
       otherwise -> Stage 2 (XGBoost)."""

    def __init__(self, min_flag_prec=0.80, min_clear_acc=0.80, max_missed_frac=0.02,
                 n_splits=20, pass_frac=0.90, random_state=42):
        self.min_flag_prec = min_flag_prec      # Stage 1 ke fraud calls: 10 mein se 8 sahi
        self.min_clear_acc = min_clear_acc      # Stage 1 ke genuine calls: 10 mein se 8 sahi
        self.max_missed_frac = max_missed_frac  # early clear hue frauds ki limit
        self.n_splits = n_splits                # kitne random 80/20 splits
        self.pass_frac = pass_frac              # kitne % splits mein rules pass hon
        self.rs = random_state
        self.low_grid = [0.0, .001, .005, .01, .02, .05, .10, .20, .30, .50]
        self.high_grid = [.50, .60, .70, .80, .90, .95, .99, .999, 1.0]

    def _lr(self):
        return make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=self.rs))

    def _xgb(self, yy):
        pos = int(yy.sum()); neg = len(yy) - pos
        return XGBClassifier(n_estimators=100, max_depth=4,
                             scale_pos_weight=neg / max(pos, 1),
                             eval_metric="logloss", random_state=self.rs, n_jobs=-1)

    def fit(self, Xt, yt):
        yt = np.asarray(yt).astype(int)
        sss = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.2,
                                     random_state=self.rs)
        rows = []
        for k, (tr, ev) in enumerate(sss.split(np.zeros(len(yt)), yt)):
            Xtr, Xev, ytr, ye = _take(Xt, tr), _take(Xt, ev), yt[tr], yt[ev]
            lr, xgb = self._lr().fit(Xtr, ytr), self._xgb(ytr).fit(Xtr, ytr)
            p1, xp = lr.predict_proba(Xev)[:, 1], xgb.predict(Xev)
            pos_e = max(int(ye.sum()), 1)
            for tl in self.low_grid:
                for th in self.high_grid:
                    if tl >= th:
                        continue
                    cleared, flagged = p1 < tl, p1 > th
                    gray = ~(cleared | flagged)
                    n_fl, n_cl = int(flagged.sum()), int(cleared.sum())
                    pred = np.where(flagged, 1, np.where(cleared, 0, xp))
                    rows.append({
                        "split": k, "T_low": tl, "T_high": th,
                        "flag_prec": ((flagged & (ye == 1)).sum() / n_fl) if n_fl else 1.0,
                        "clear_acc": ((cleared & (ye == 0)).sum() / n_cl) if n_cl else 1.0,
                        "missed_frac": (cleared & (ye == 1)).sum() / pos_e,
                        "gray_pct": gray.mean() * 100,
                        "n_flagged": n_fl,
                        "recall": ((pred == 1) & (ye == 1)).sum() / pos_e})
            print(f"  tuning split {k+1}/{self.n_splits} done", end="\r")

        df_ = pd.DataFrame(rows)
        df_["ok"] = ((df_.flag_prec >= self.min_flag_prec) &
                     (df_.clear_acc >= self.min_clear_acc) &
                     (df_.missed_frac <= self.max_missed_frac))
        agg = (df_.groupby(["T_low", "T_high"])
                  .agg(pass_rate=("ok", "mean"), gray_pct=("gray_pct", "mean"),
                       flag_prec_mean=("flag_prec", "mean"),
                       clear_acc_mean=("clear_acc", "mean"),
                       missed_frac_mean=("missed_frac", "mean"),
                       recall_mean=("recall", "mean"),
                       avg_flagged=("n_flagged", "mean"))
                  .reset_index())

        cand = agg[agg.pass_rate >= self.pass_frac]
        if len(cand):
            best = cand.sort_values(["gray_pct", "recall_mean"],
                                    ascending=[True, False]).iloc[0]
            self.t_low_, self.t_high_ = float(best["T_low"]), float(best["T_high"])
            self.chosen_, self.fallback_ = best, False
        else:                                   # koi combo stable nahi -> sab XGBoost ko
            self.t_low_, self.t_high_, self.chosen_, self.fallback_ = 0.0, 1.0, None, True

        self.tuning_table_ = agg
        self.lr_, self.xgb_ = self._lr().fit(Xt, yt), self._xgb(yt).fit(Xt, yt)
        return self

    def predict(self, Xn, return_gray=False):
        p1 = self.lr_.predict_proba(Xn)[:, 1]
        pred = np.zeros(len(p1), dtype=int)
        pred[p1 > self.t_high_] = 1
        gray = np.where((p1 >= self.t_low_) & (p1 <= self.t_high_))[0]
        if len(gray):
            pred[gray] = self.xgb_.predict(_take(Xn, gray))
        return (pred, gray) if return_gray else pred

    def report(self, Xte, yte, runs=10):
        yv = np.asarray(yte).astype(int)
        p1 = self.lr_.predict_proba(Xte)[:, 1]
        cleared, flagged = p1 < self.t_low_, p1 > self.t_high_
        gray = ~(cleared | flagged)

        print("\n" + "=" * 62)
        print("                 FINAL TEST-SET REPORT")
        print("=" * 62)
        print(f"T_low = {self.t_low_:.4f} | T_high = {self.t_high_:.4f}"
              + ("   [FALLBACK: no stable combo, all data goes to XGBoost]"
                 if self.fallback_ else ""))
        if self.chosen_ is not None:
            print(f"Rules passed in {self.chosen_['pass_rate']*100:.0f}% of "
                  f"{self.n_splits} random 80/20 splits")

        n_c, n_f = int(cleared.sum()), int(flagged.sum())
        print(f"\nStage 1 cleared as genuine : {n_c} | wrongly cleared (were fraud): "
              f"{int((cleared & (yv == 1)).sum())}"
              + (f" | accuracy {((cleared & (yv == 0)).sum() / n_c) * 100:.2f}%" if n_c else ""))
        print(f"Stage 1 flagged as fraud   : {n_f} | wrongly flagged (were genuine): "
              f"{int((flagged & (yv == 0)).sum())}"
              + (f" | precision {((flagged & (yv == 1)).sum() / n_f) * 100:.1f}%" if n_f else ""))
        print(f"Sent to XGBoost            : {int(gray.sum())} / {len(yv)} "
              f"({gray.mean() * 100:.2f}%)  -> XGBoost load reduced by {100 - gray.mean() * 100:.2f}%")

        pred, base = self.predict(Xte), self.xgb_.predict(Xte)
        print("\nCASCADE vs SINGLE XGBOOST (same test set):")
        for name, p in [("Single XGBoost", base), ("Cascade", pred)]:
            tp = int(((p == 1) & (yv == 1)).sum()); fp = int(((p == 1) & (yv == 0)).sum())
            fn = int(((p == 0) & (yv == 1)).sum())
            print(f"  {name:15s}: recall={tp/(tp+fn):.3f} | precision={tp/max(tp+fp,1):.3f} | "
                  f"frauds caught={tp}/{tp+fn} | false alarms={fp}")

        def timeit(fn):
            fn()                                 # warm-up
            ts = []
            for _ in range(runs):
                s = time.perf_counter(); fn(); ts.append(time.perf_counter() - s)
            return float(np.median(ts))
        t_c, t_s = timeit(lambda: self.predict(Xte)), timeit(lambda: self.xgb_.predict(Xte))
        print(f"\nTIMING (median of {runs} runs): single XGBoost {t_s*1000:.1f} ms | "
              f"cascade {t_c*1000:.1f} ms | change {(1 - t_c/t_s)*100:.1f}% "
              f"(negative = cascade slower)")
        print("=" * 62)


# ------------------------------------------------------------------------------
# STEP 3: TUNE ON TRAIN ONLY, THEN REPORT ON TEST ONCE
# ------------------------------------------------------------------------------
print("\n--- Tuning thresholds on training data (20 random 80/20 splits) ---")
t0 = time.time()
model = RobustCascade(n_splits=20).fit(X_train, y_train)
print(f"\nTuning + final training took {time.time() - t0:.0f} s")

model.report(X_test, y_test)

print("\nTop 10 threshold combinations (lowest XGBoost load first):")
print(model.tuning_table_.sort_values(["pass_rate", "gray_pct"],
                                      ascending=[False, True]).head(10).to_string(index=False))
