import re
import pandas as pd
 

# Cấu hình

INPUT_FILE  = "1du_lieu_sach.csv"
OUTPUT_FILE = "2du_lieu_FE.csv"
 

# Từ điển đặc trưng theo chủ đề lừa đảo

SCAM_KEYWORDS = [
    "chuyển khoản", "nộp phí", "đặt cọc", "click link", "link lạ",
    "trúng thưởng", "khẩn cấp", "ngay lập tức", "tuyệt mật",
    "không được tiết lộ", "lãi suất 0%", "việc nhẹ lương cao",
    "lợi nhuận", "hoa hồng", "nạp tiền", "rửa tiền", "phong tỏa",
    "khởi tố", "bắt giữ", "phí hành chính", "bằng chứng",
    "thông tin cá nhân", "bị hủy", "cơ hội vàng", "siêu rẻ",
    "mượn tiền", "tài khoản giám sát", "tống tiền",
]
 
CLEAN_KEYWORDS = [
    "kênh chính thức", "website chính thức", "liên hệ trực tiếp",
    "không thu phí", "không yêu cầu", "thông tin chính thống",
    "phòng đào tạo", "phòng công tác sinh viên",
    "hus.vnu.edu.vn", "vnu.edu.vn",
]
 
URGENT_PATTERN  = re.compile(r"ngay|gấp|khẩn|hôm nay|lập tức|trong vòng", re.IGNORECASE)
THREAT_PATTERN  = re.compile(r"bị bắt|khởi tố|phong tỏa|tuyệt mật|không tiết lộ", re.IGNORECASE)
MONEY_PATTERN   = re.compile(r"triệu|vnđ|vnd|đồng|\d+\.000", re.IGNORECASE)
LINK_PATTERN    = re.compile(r"<url>|bit\.ly|tinyurl|link lạ|link giả", re.IGNORECASE)
PHONE_PATTERN   = re.compile(r"<phone>", re.IGNORECASE)
ACCT_PATTERN    = re.compile(r"<account>", re.IGNORECASE)
IMPERSON_PATTERN= re.compile(r"công an|tòa án|ngân hàng|viện kiểm sát|hải quan", re.IGNORECASE)
SECRET_PATTERN  = re.compile(r"tuyệt mật|bí mật|không kể|không tiết lộ|riêng tư", re.IGNORECASE)
REWARD_PATTERN  = re.compile(r"trúng thưởng|trúng tuyển|học bổng|quà tặng|phần thưởng", re.IGNORECASE)
INVEST_PATTERN  = re.compile(r"đầu tư|lợi nhuận|hoa hồng|lãi suất|sinh lời|nạp tiền", re.IGNORECASE)
 
 

# Hàm trích xuất đặc trưng thống kê

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm 14 cột đặc trưng thống kê vào DataFrame.
    Tất cả tính trên cột Sub_Content_clean (đã viết thường).
    """
    txt = df["Sub_Content_clean"]
 
    # Đặc trưng độ dài
    df["feat_char_count"]  = txt.str.len()
    df["feat_word_count"]  = txt.str.split().str.len()
    df["feat_exclaim"]     = txt.str.count(r"!")
    df["feat_question"]    = txt.str.count(r"\?")
 
    # Đặc trưng token placeholder (từ bước làm sạch) 
    df["feat_has_url"]     = txt.str.contains("<url>",     regex=False).astype(int)
    df["feat_has_phone"]   = txt.str.contains("<phone>",   regex=False).astype(int)
    df["feat_has_account"] = txt.str.contains("<account>", regex=False).astype(int)
    df["feat_has_money"]   = txt.str.contains("<money>",   regex=False).astype(int)
 
    # Từ khoá chủ đề 
    df["feat_scam_kw_count"]  = txt.apply(
        lambda t: sum(1 for kw in SCAM_KEYWORDS if kw in t)
    )
    df["feat_clean_kw_count"] = txt.apply(
        lambda t: sum(1 for kw in CLEAN_KEYWORDS if kw in t)
    )
 
    # Mẫu hành vi lừa đảo điển hình
    df["feat_urgent"]     = txt.apply(lambda t: int(bool(URGENT_PATTERN.search(t))))
    df["feat_threat"]     = txt.apply(lambda t: int(bool(THREAT_PATTERN.search(t))))
    df["feat_impersonate"]= txt.apply(lambda t: int(bool(IMPERSON_PATTERN.search(t))))
    df["feat_invest"]     = txt.apply(lambda t: int(bool(INVEST_PATTERN.search(t))))
 
    return df
 
 
def encode_label(df: pd.DataFrame) -> pd.DataFrame:
    """Mã hoá nhãn: SCAM → 1, CLEAN → 0 → cột label_enc."""
    df["label_enc"] = df["Label"].map({"SCAM": 1, "CLEAN": 0})
    return df
 
 
def run(input_file: str = INPUT_FILE, output_file: str = OUTPUT_FILE) -> pd.DataFrame:
    print("=" * 55)
    print("  BƯỚC 2: FEATURE ENGINEERING")
    print("=" * 55)
 
    print(f"\n[1/3] Đọc {input_file} ...")
    df = pd.read_csv(input_file, encoding="utf-8")
    df["Sub_Content_clean"] = df["Sub_Content_clean"].fillna("")
    print(f"  → {len(df):,} dòng")
 
    print("\n[2/3] Trích xuất 14 đặc trưng thống kê ...")
    df = extract_features(df)
    df = encode_label(df)
 
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    print(f"  → {len(feat_cols)} đặc trưng:")
    for fc in feat_cols:
        print(f"     • {fc}: mean={df[fc].mean():.3f}  max={df[fc].max()}")
 
    print(f"\n[3/3] Lưu → {output_file}")
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"  → {len(df):,} dòng, {len(df.columns)} cột")
    print("=" * 55)
    return df
 
 
if __name__ == "__main__":
    run()