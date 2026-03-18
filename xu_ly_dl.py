import re
import pandas as pd
 

# Cấu hình
INPUT_FILE  = "data.csv"
OUTPUT_FILE = "1du_lieu_sach.csv"
 
# Regex chuẩn hoá token đặc biệt
_URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b0\d{9,10}\b")
_ACCT_RE  = re.compile(r"\b\d{8,20}\b")
_MONEY_RE = re.compile(r"\d[\d\.,]*\s*(triệu|nghìn|vnđ|vnd|đồng)", re.IGNORECASE)
_MULTI_SP = re.compile(r"\s{2,}")
 
 
def clean_text(text: str) -> str:
    """
    Pipeline làm sạch 1 chuỗi văn bản:
      1. Loại ký tự điều khiển
      2. URL        → <URL>
      3. Số tiền    → <MONEY>
      4. Số tài khoản → <ACCOUNT>
      5. Số điện thoại → <PHONE>
      6. Loại ký tự đặc biệt thừa
      7. Chuẩn hoá khoảng trắng, viết thường
    """
    if not isinstance(text, str):
        return ""
    text = "".join(ch for ch in text if ch >= " " or ch in "\n\t")
    text = _URL_RE.sub(" <URL> ", text)
    text = _MONEY_RE.sub(" <MONEY> ", text)
    text = _ACCT_RE.sub(" <ACCOUNT> ", text)
    text = _PHONE_RE.sub(" <PHONE> ", text)
    text = re.sub(r"[^\w\s\u00C0-\u024F\u1E00-\u1EFF<>.,!?;:()\-]", " ", text)
    text = _MULTI_SP.sub(" ", text).strip().lower()
    return text
 
 
def load_raw(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding="utf-8")
    required = {"Sub_Content", "Label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"File CSV thiếu cột: {missing}")
    return df
 
 
def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["Sub_Content", "Label"])
    df = df[df["Sub_Content"].str.strip() != ""]
    df["Label"] = df["Label"].str.strip().str.upper()
    df = df[df["Label"].isin(["SCAM", "CLEAN"])]
    after = len(df)
    if before != after:
        print(f"  [!] Loại {before - after} dòng không hợp lệ/NaN (còn {after:,})")
    return df.reset_index(drop=True)
 
 
def run(input_file: str = INPUT_FILE, output_file: str = OUTPUT_FILE) -> pd.DataFrame:
    print("=" * 55)
    print("  BƯỚC 1: LÀM SẠCH DỮ LIỆU THÔ")
    print("=" * 55)
 
    print(f"\n[1/3] Đọc {input_file} ...")
    df = load_raw(input_file)
    print(f"  → {len(df):,} dòng, {len(df.columns)} cột")
 
    print("\n[2/3] Làm sạch văn bản ...")
    df = remove_invalid_rows(df)
    df["Sub_Content_clean"] = df["Sub_Content"].apply(clean_text)
    df["Reason"] = df["Reason"].fillna("").str.strip()
 
    counts = df["Label"].value_counts()
    total  = len(df)
    print(f"\n  Phân bố nhãn:")
    for lbl, cnt in counts.items():
        bar = "█" * int(cnt / total * 25)
        print(f"    {lbl:6s}  {cnt:4d} ({cnt/total*100:.1f}%)  {bar}")
 
    print(f"\n[3/3] Lưu → {output_file}")
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"  → {len(df):,} dòng, {len(df.columns)} cột")
    print("=" * 55)
    return df
 
 
if __name__ == "__main__":
    run()