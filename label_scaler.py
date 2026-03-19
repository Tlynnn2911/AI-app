import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix, save_npz
 

# Cấu hình
INPUT_FILE      = "2du_lieu_FE.csv"
OUTPUT_REF_CSV  = "3du_lieu_da_ma_hoa.csv"   # CSV tham chiếu (không dùng cho training)
VEC_FILE        = "vectorizer.pkl"
SCALER_FILE     = "scaler.pkl"
FEAT_COLS_FILE  = "feature_columns.pkl"
 
# Cột đặc trưng thống kê (14 cột từ xu_ly_fe.py)
STAT_FEAT_COLS = [
    "feat_char_count",
    "feat_word_count",
    "feat_exclaim",
    "feat_question",
    "feat_has_url",
    "feat_has_phone",
    "feat_has_account",
    "feat_has_money",
    "feat_scam_kw_count",
    "feat_clean_kw_count",
    "feat_urgent",
    "feat_threat",
    "feat_impersonate",
    "feat_invest",
]
 
 
def build_vectorizers(texts: pd.Series) -> dict:
    """
    Khởi tạo và fit 2 TF-IDF vectorizer:
      - vec_word: word n-gram (1,2)  — bắt cụm từ quan trọng
      - vec_char: char_wb n-gram (2,4) — bắt biến thể chính tả
    """
    vec_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=25_000,
        sublinear_tf=True,
        min_df=2,
    )
    vec_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=15_000,
        sublinear_tf=True,
        min_df=3,
    )
    vec_word.fit(texts)
    vec_char.fit(texts)
    return {"word": vec_word, "char": vec_char}
 
 
def build_scaler(df: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler trên toàn bộ đặc trưng thống kê."""
    scaler = StandardScaler()
    scaler.fit(df[STAT_FEAT_COLS].values.astype(float))
    return scaler
 
 
def transform_all(df: pd.DataFrame, vectorizer: dict, scaler: StandardScaler):
    """Ghép TF-IDF (word + char) + đặc trưng thống kê đã chuẩn hoá → sparse matrix."""
    X_word = vectorizer["word"].transform(df["Sub_Content_clean"])
    X_char = vectorizer["char"].transform(df["Sub_Content_clean"])
    X_stat = csr_matrix(
        scaler.transform(df[STAT_FEAT_COLS].values.astype(float))
    )
    return hstack([X_word, X_char, X_stat])
 
 
def run(input_file: str = INPUT_FILE) -> tuple:
    print("=" * 55)
    print("  BƯỚC 3: VECTOR HOÁ & CHUẨN HOÁ ĐẶC TRƯNG")
    print("=" * 55)
 
    # 1. Đọc dữ liệu FE
    print(f"\n[1/4] Đọc {input_file} ...")
    df = pd.read_csv(input_file, encoding="utf-8")
    df["Sub_Content_clean"] = df["Sub_Content_clean"].fillna("")
    for col in STAT_FEAT_COLS:
        df[col] = df[col].fillna(0)
    print(f"  → {len(df):,} mẫu")
 
    # 2. Fit vectorizer TF-IDF
    print("\n[2/4] Fit TF-IDF vectorizer ...")
    vectorizer = build_vectorizers(df["Sub_Content_clean"])
    vocab_word = len(vectorizer["word"].vocabulary_)
    vocab_char = len(vectorizer["char"].vocabulary_)
    print(f"  → TF-IDF word vocab: {vocab_word:,}")
    print(f"  → TF-IDF char vocab: {vocab_char:,}")
 
    # 3. Fit scaler cho đặc trưng thống kê
    print("\n[3/4] Fit StandardScaler cho 14 đặc trưng thống kê ...")
    scaler = build_scaler(df)
    print(f"  → Scale {len(STAT_FEAT_COLS)} cột: {STAT_FEAT_COLS}")
 
    # 4. Lưu artifacts
    print("\n[4/4] Lưu artifacts ...")
    joblib.dump(vectorizer,    VEC_FILE)
    joblib.dump(scaler,        SCALER_FILE)
    joblib.dump(STAT_FEAT_COLS, FEAT_COLS_FILE)
    print(f"   {VEC_FILE}")
    print(f"   {SCALER_FILE}")
    print(f"   {FEAT_COLS_FILE}")
 
    # Lưu CSV tham chiếu (chỉ các cột số, không chứa sparse matrix)
    df_ref = df[["Sub_Content", "Label", "label_enc"] + STAT_FEAT_COLS].copy()
    df_ref.to_csv(OUTPUT_REF_CSV, index=False, encoding="utf-8")
    print(f"  {OUTPUT_REF_CSV}  (CSV tham chiếu)")
 
    print("\n" + "=" * 55)
    print(f" Tổng chiều đặc trưng: {vocab_word + vocab_char + len(STAT_FEAT_COLS):,}")
    print("=" * 55)
    return vectorizer, scaler
 
 
if __name__ == "__main__":
    run()