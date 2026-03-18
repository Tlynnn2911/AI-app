import time
import joblib
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB  # noqa (kept for reference)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)
 
warnings.filterwarnings("ignore")
 

# Cấu hình
DATA_FILE       = "2du_lieu_FE.csv"
VEC_FILE        = "vectorizer.pkl"
SCALER_FILE     = "scaler.pkl"
FEAT_COLS_FILE  = "feature_columns.pkl"
MODEL_FILE      = "model.pkl"
REPORT_FILE     = "bao_cao_huan_luyen.txt"
RANDOM_SEED     = 42
TEST_SIZE       = 0.2
N_FOLDS         = 5
 
 

# 1. Tải dữ liệu & artifacts từ bước trước
def load_all():
    df          = pd.read_csv(DATA_FILE, encoding="utf-8")
    vectorizer  = joblib.load(VEC_FILE)
    scaler      = joblib.load(SCALER_FILE)
    feat_cols   = joblib.load(FEAT_COLS_FILE)
 
    df["Sub_Content_clean"] = df["Sub_Content_clean"].fillna("")
    for col in feat_cols:
        df[col] = df[col].fillna(0)
 
    return df, vectorizer, scaler, feat_cols
 
 

# 2. Xây dựng ma trận đặc trưng kết hợp
def build_X(df: pd.DataFrame, vectorizer: dict, scaler, feat_cols: list):
    X_word = vectorizer["word"].transform(df["Sub_Content_clean"])
    X_char = vectorizer["char"].transform(df["Sub_Content_clean"])
    X_stat = csr_matrix(
        scaler.transform(df[feat_cols].values.astype(float))
    )
    return hstack([X_word, X_char, X_stat])
 
 
# 3. Định nghĩa các mô hình
def build_models() -> dict:
    """
    Lưu ý: Naive Bayes (kể cả ComplementNB) không tương thích với ma trận
    hỗn hợp TF-IDF + StandardScaler (sinh giá trị âm).
    Dùng Logistic Regression, Random Forest và Ensemble LR+RF.
    """
    lr = LogisticRegression(
        C=5.0, max_iter=1000, solver="lbfgs",
        class_weight="balanced", random_state=RANDOM_SEED,
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
    )
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf)],
        voting="soft",
        weights=[3, 2],
    )
    return {
        "Logistic Regression": lr,
        "Random Forest":       rf,
        "Ensemble (LR+RF)":    ensemble,
    }
 
 

# 4. Cross-validation tất cả mô hình
def cross_validate(models: dict, X_train, y_train) -> dict:
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results = {}
    print(f"\n  Cross-validation ({N_FOLDS}-fold):")
    print(f"  {'Mô hình':<25} {'Acc':>7}  {'F1':>7}  {'AUC':>7}  {'Thời gian':>10}")
    print(f"  {'─'*65}")
 
    for name, model in models.items():
        t0  = time.time()
        acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy",  n_jobs=-1)
        f1  = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro",  n_jobs=-1)
        auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc",   n_jobs=-1)
        elapsed = time.time() - t0
 
        results[name] = {
            "acc": acc.mean(), "acc_std": acc.std(),
            "f1":  f1.mean(),  "f1_std":  f1.std(),
            "auc": auc.mean(), "auc_std": auc.std(),
        }
        print(f"  {name:<25} {acc.mean():.4f}   {f1.mean():.4f}   {auc.mean():.4f}   {elapsed:>6.1f}s")
 
    return results
 
 
# 5. Huấn luyện mô hình tốt nhất trên toàn train set
def train_best(models: dict, cv_results: dict, X_train, y_train, X_test, y_test) -> tuple:
    best_name  = max(cv_results, key=lambda k: cv_results[k]["f1"])
    best_model = models[best_name]
    print(f"\n  Mô hình tốt nhất (F1 cao nhất): {best_name}")
    print(f"     Đang huấn luyện trên toàn bộ tập train ...")
    best_model.fit(X_train, y_train)
 
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
 
    metrics = {
        "model_name": best_name,
        "accuracy":   accuracy_score(y_test, y_pred),
        "auc":        roc_auc_score(y_test, y_prob),
        "report":     classification_report(
            y_test, y_pred, target_names=["CLEAN", "SCAM"], digits=4
        ),
        "confusion":  confusion_matrix(y_test, y_pred),
    }
    return best_model, metrics
 
 
# 6. In & lưu báo cáo
def save_report(metrics: dict, cv_results: dict) -> str:
    lines = [
        "=" * 62,
        "  BÁO CÁO HUẤN LUYỆN — ANTISCAM DETECTOR",
        "=" * 62,
        f"\n  Mô hình được chọn : {metrics['model_name']}",
        f"  Accuracy (test)   : {metrics['accuracy']:.4f}",
        f"  AUC-ROC  (test)   : {metrics['auc']:.4f}",
        f"\n{'─'*62}",
        "  Classification Report:",
        metrics["report"],
        f"{'─'*62}",
        "  Confusion Matrix:",
        f"              Pred CLEAN   Pred SCAM",
    ]
    cm = metrics["confusion"]
    lines.append(f"  True CLEAN     {cm[0,0]:6d}      {cm[0,1]:6d}")
    lines.append(f"  True SCAM      {cm[1,0]:6d}      {cm[1,1]:6d}")
    lines.append(f"\n{'─'*62}")
    lines.append("  Kết quả Cross-Validation (toàn bộ):")
    lines.append(f"  {'Mô hình':<25} {'Acc':>7}  {'F1':>7}  {'AUC':>7}")
    for name, r in cv_results.items():
        lines.append(f"  {name:<25} {r['acc']:.4f}   {r['f1']:.4f}   {r['auc']:.4f}")
    lines.append("=" * 62)
 
    text = "\n".join(lines)
    print("\n" + text)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n Báo cáo → {REPORT_FILE}")
    return text
 
 
# Pipeline chính
def run():
    print("=" * 62)
    print("  BƯỚC 4: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 62)
 
    # 1. Tải dữ liệu & artifacts
    print(f"\n[1/5] Tải dữ liệu và artifacts ...")
    df, vectorizer, scaler, feat_cols = load_all()
    y = df["label_enc"].values
    print(f"  → {len(df):,} mẫu  |  SCAM={y.sum()}  CLEAN={(y==0).sum()}")
 
    # 2. Chia train/test (stratified)
    print(f"\n[2/5] Chia train/test ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}, stratified) ...")
    idx_train, idx_test = train_test_split(
        range(len(df)), test_size=TEST_SIZE,
        random_state=RANDOM_SEED, stratify=y,
    )
    df_train, df_test = df.iloc[idx_train], df.iloc[idx_test]
    y_train = df_train["label_enc"].values
    y_test  = df_test["label_enc"].values
    print(f"  → Train: {len(df_train):,}  |  Test: {len(df_test):,}")
 
    # 3. Xây dựng ma trận đặc trưng
    print("\n[3/5] Xây dựng ma trận đặc trưng [TF-IDF + stat] ...")
    X_train = build_X(df_train, vectorizer, scaler, feat_cols)
    X_test  = build_X(df_test,  vectorizer, scaler, feat_cols)
    print(f"  → X_train: {X_train.shape}  |  X_test: {X_test.shape}")
 
    # 4. Cross-validate
    print("\n[4/5] Đánh giá cross-validation tất cả mô hình ...")
    models     = build_models()
    cv_results = cross_validate(models, X_train, y_train)
 
    # 5. Huấn luyện & lưu model tốt nhất
    print("\n[5/5] Huấn luyện mô hình tốt nhất & đánh giá tập test ...")
    best_model, metrics = train_best(models, cv_results, X_train, y_train, X_test, y_test)
 
    bundle = {
        "model":      best_model,
        "feat_cols":  feat_cols,
        "model_name": metrics["model_name"],
    }
    joblib.dump(bundle, MODEL_FILE)
    print(f" Model → {MODEL_FILE}")
 
    save_report(metrics, cv_results)
    return best_model
 
 
if __name__ == "__main__":
    run()
 