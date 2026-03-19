import re
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify
from scipy.sparse import hstack, csr_matrix
 
import xu_ly_dl as xldl
import xu_ly_fe  as xfe
 
 
# Cấu hình
 
MODEL_FILE     = "model.pkl"
VEC_FILE       = "vectorizer.pkl"
SCALER_FILE    = "scaler.pkl"
FEAT_COLS_FILE = "feature_columns.pkl"
DATA_FILE      = "data.csv"
THRESHOLD      = 0.50
 
app = Flask(__name__)
 
# Tải mô hình khi khởi động
 
def load_artifacts():
    """Tải model + vectorizer + scaler; tự huấn luyện nếu chưa có."""
    missing = [f for f in [MODEL_FILE, VEC_FILE, SCALER_FILE, FEAT_COLS_FILE]
               if not os.path.exists(f)]
    if missing:
        print(f"  [!] Thiếu artifacts: {missing} — chạy pipeline huấn luyện ...")
        import xu_ly_dl, xu_ly_fe, label_scaler, huan_luyen
        xu_ly_dl.run()
        xu_ly_fe.run()
        label_scaler.run()
        huan_luyen.run()
 
    bundle     = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VEC_FILE)
    scaler     = joblib.load(SCALER_FILE)
    feat_cols  = joblib.load(FEAT_COLS_FILE)
    return bundle["model"], vectorizer, scaler, feat_cols
 
 
model, vectorizer, scaler, feat_cols = load_artifacts()
print(f" Model sẵn sàng: {type(model).__name__}")
 
# Tải dữ liệu mẫu để hiển thị lịch sử / autocomplete
try:
    df_data = pd.read_csv(DATA_FILE, encoding="utf-8").dropna(subset=["Sub_Content", "Label"])
    df_data["Label"] = df_data["Label"].str.strip().str.upper()
except Exception:
    df_data = pd.DataFrame(columns=["Sub_Content", "Label", "Reason"])
 
 
# Tiền xử lý & dự đoán
 
def preprocess_one(text: str):
    """Làm sạch → vector hoá → sparse matrix 1 dòng."""
    cleaned = xldl.clean_text(text)
 
    # Tạo DataFrame tạm để extract_features hoạt động đúng
    row = pd.DataFrame([{
        "Sub_Content":       text,
        "Sub_Content_clean": cleaned,
        "Label":             "SCAM",
        "label_enc":         1,
    }])
    row = xfe.extract_features(row)
 
    # Đảm bảo đủ các cột stat
    for col in feat_cols:
        if col not in row.columns:
            row[col] = 0
 
    X_word = vectorizer["word"].transform([cleaned])
    X_char = vectorizer["char"].transform([cleaned])
    X_stat = csr_matrix(
        scaler.transform(row[feat_cols].values.astype(float))
    )
    return hstack([X_word, X_char, X_stat])
 
 
def predict_text(text: str, threshold: float = THRESHOLD) -> dict:
    """
    Dự đoán 1 văn bản.
    Trả về: label, scam_prob, clean_prob, confidence, signals
    """
    X         = preprocess_one(text)
    proba     = model.predict_proba(X)[0]          # [P(CLEAN), P(SCAM)]
    scam_prob = float(proba[1])
    label     = "SCAM" if scam_prob >= threshold else "CLEAN"
    confidence= scam_prob if label == "SCAM" else 1.0 - scam_prob
 
    signals = _detect_signals(text)
    return {
        "label":      label,
        "scam_prob":  round(scam_prob * 100, 1),
        "clean_prob": round((1 - scam_prob) * 100, 1),
        "confidence": round(confidence * 100, 1),
        "signals":    signals,
    }
 
 
# Phát hiện dấu hiệu lừa đảo cụ thể
 
_SIGNAL_CHECKS = [
    (r"chuyển khoản|nộp phí|đặt cọc",              " Yêu cầu chuyển tiền / nộp phí"),
    (r"click link|link lạ|link giả|bit\.ly",        " Chứa đường link đáng ngờ"),
    (r"ngay lập tức|khẩn cấp|gấp|hôm nay",         " Tạo áp lực thời gian khẩn cấp"),
    (r"tuyệt mật|không được tiết lộ|bí mật",        " Yêu cầu giữ bí mật"),
    (r"công an|tòa án|viện kiểm sát|hải quan",      " Giả danh cơ quan chức năng"),
    (r"lãi suất 0%|không lãi suất",                 " Mời vay lãi suất 0% không thực tế"),
    (r"việc nhẹ lương cao|thu nhập.*triệu.*ngày",   " Việc nhẹ lương cao bất thường"),
    (r"lợi nhuận.*%|cam kết hoàn vốn|nạp tiền",    " Đầu tư lợi nhuận phi thực tế"),
    (r"\bcccd\b|\bcmnd\b|căn cước",                 " Thu thập giấy tờ tuỳ thân"),
    (r"rửa tiền|phong tỏa|khởi tố|bắt giữ",        " Đe dọa pháp lý"),
    (r"trúng thưởng|trúng tuyển|học bổng.*phí",     " Thông báo trúng thưởng / học bổng giả"),
    (r"hải quan.*phí|phí vận chuyển",               " Giả mạo phí hải quan / vận chuyển"),
    (r"đầu tư|lợi nhuận|hoa hồng|lãi suất|sinh lời", " Mời gọi đầu tư sinh lời bất thường"),
    (r"vay|tiền|ck|chuyển khoản|mai trả|mượn tiền", " Lừa đảo chuyển tiền / cho vay"),
    (r"thử ngay|ấn ngay|thu ngay|an ngay|lien he|liên hệ", "Đường link / lời kêu gọi hành động đáng ngờ"),
]
 
def _detect_signals(text: str) -> list:
    signals = []
    lower   = text.lower()
    for pattern, msg in _SIGNAL_CHECKS:
        if re.search(pattern, lower):
            signals.append(msg)
    if not signals:
        signals.append(" Không phát hiện dấu hiệu lừa đảo rõ ràng")
    return signals
 
 
# Giao diện HTML (inline template)
 
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AntiScam Detector</title>
<style>
  :root {
    --navy-dark:  #2D4A7A;
    --navy:       #35578F;
    --navy-mid:   #3558A8;
    --navy-light: #3D66BF;
    --navy-card:  #2A4585;
    --navy-row:   #9acbff;
    --orange:     #F5A623;
    --orange2:    #E8941A;
    --orange3:    #D4820F;
    --orange-bg:  rgba(245,166,35,.12);
    --orange-bd:  rgba(245,166,35,.35);
    --text:       #FFFFFF;
    --text-dim:   rgba(255,255,255,.7);
    --text-muted: rgba(255,255,255,.42);
    --border:     rgba(255,255,255,.1);
    --border2:    rgba(255,255,255,.18);
    --red:        #FF6B6B;
    --red-bg:     rgba(255,107,107,.14);
    --red-bd:     rgba(255,107,107,.35);
    --green:      #4ADE80;
    --green-bg:   rgba(74,222,128,.12);
    --green-bd:   rgba(74,222,128,.35);
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--navy-dark);
    color: var(--text);
    min-height: 100vh;
  }
 
  /* ── HEADER ── */
  header {
    background: linear-gradient(135deg, #1E2F58 0%, #2A4585 60%, #3558A8 100%);
    border-bottom: 2px solid var(--orange);
    padding: .875rem 2rem;
    display: flex; align-items: center; gap: 1rem;
  }
  .logo { font-size: 1.4rem; font-weight: 900; letter-spacing: .04em; color: #fff; }
  .logo span { color: var(--orange); }
  .logo-sub { font-size: .65rem; color: var(--text-muted); letter-spacing: .12em; text-transform: uppercase; margin-top: 1px; }
  .badge {
    padding: 5px 14px;
    background: var(--orange-bg);
    border: 1px solid var(--orange-bd);
    border-radius: 100px;
    font-size: .72rem; color: var(--orange);
    display: flex; align-items: center; gap: 6px;
    margin-left: auto; font-weight: 600;
  }
  .dot { width: 6px; height: 6px; background: var(--orange); border-radius: 50%; animation: blink 1.5s infinite; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }
 
  /* ── LAYOUT ── */
  main { max-width: 1120px; margin: 0 auto; padding: 1.75rem 1.5rem; }
  .grid { display: grid; grid-template-columns: 1fr 390px; gap: 1.5rem; align-items: start; }
  @media(max-width:900px){ .grid{ grid-template-columns:1fr; } }
 
  /* ── CARDS ── */
  .card {
    background: var(--navy-card);
    border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden;
  }
  .card-header {
    padding: .875rem 1.25rem;
    border-bottom: 1px solid var(--border);
    font-size: .72rem; font-weight: 700;
    color: var(--orange);
    text-transform: uppercase; letter-spacing: .1em;
    display: flex; justify-content: space-between; align-items: center;
    background: rgba(0,0,0,.2);
  }
 
  /* ── TEXTAREA ── */
  textarea {
    width: 100%; background: transparent; border: none; outline: none;
    color: var(--text); font-family: inherit; font-size: .9375rem;
    line-height: 1.7; padding: 1.25rem; resize: none; min-height: 200px;
  }
  textarea::placeholder { color: var(--text-muted); }
  @keyframes highlightPulse {
    0%   { box-shadow: 0 0 0 0 rgba(245,166,35,.5); }
    70%  { box-shadow: 0 0 0 10px rgba(245,166,35,0); }
    100% { box-shadow: 0 0 0 0 rgba(245,166,35,0); }
  }
  .highlight-pulse { animation: highlightPulse .8s ease; }
 
  /* ── FOOTER ── */
  .card-footer {
    padding: .875rem 1.25rem;
    border-top: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    gap: 1rem; flex-wrap: wrap;
    background: rgba(0,0,0,.15);
  }
  .chips { display: flex; gap: 6px; flex-wrap: wrap; }
  .chip {
    padding: 4px 10px;
    background: rgba(255,255,255,.06);
    border: 1px solid var(--border2);
    border-radius: 6px; font-size: .72rem;
    color: var(--text-dim); cursor: pointer; transition: .15s;
  }
  .chip:hover { border-color: var(--orange); color: var(--orange); background: var(--orange-bg); }
 
  /* ── BUTTON ── */
  .btn {
    padding: 10px 26px;
    background: var(--orange);
    border: none; border-radius: 8px;
    color: #1A2744; font-weight: 800; font-size: .875rem;
    cursor: pointer; transition: .2s; font-family: inherit;
    letter-spacing: .02em;
  }
  .btn:hover { background: var(--orange2); transform: translateY(-1px); }
  .btn:disabled { opacity: .4; cursor: not-allowed; transform: none; }
 
  /* ── RESULT PANEL ── */
  .result-placeholder { padding: 3rem; text-align: center; color: var(--text-muted); font-size: .875rem; }
  .result-placeholder-icon { font-size: 2.5rem; margin-bottom: .75rem; opacity: .2; }
  #result-content { display: none; }
 
  .verdict { padding: 1.75rem; text-align: center; }
  .verdict.scam {
    background: var(--red-bg);
    border-bottom: 1px solid var(--red-bd);
  }
  .verdict.clean {
    background: var(--green-bg);
    border-bottom: 1px solid var(--green-bd);
  }
  .verdict-icon { font-size: 2.5rem; display: block; margin-bottom: .5rem; }
  .verdict-label { font-size: 1.75rem; font-weight: 900; letter-spacing: .06em; }
  .verdict.scam .verdict-label  { color: var(--red); }
  .verdict.clean .verdict-label { color: var(--green); }
  .verdict-sub { font-size: .8rem; color: var(--text-dim); margin-top: .25rem; }
 
  .conf-wrap { padding: 1.25rem; }
  .conf-row {
    display: flex; justify-content: space-between; margin-bottom: 6px;
    font-size: .7rem; color: var(--text-muted); font-weight: 700;
    text-transform: uppercase; letter-spacing: .07em;
  }
  .conf-bar { height: 7px; background: rgba(255,255,255,.1); border-radius: 100px; overflow: hidden; }
  .conf-fill { height: 100%; border-radius: 100px; transition: width .7s cubic-bezier(.4,0,.2,1); width: 0%; }
  .conf-fill.scam  { background: linear-gradient(90deg, #FF6B6B, #FF4444); }
  .conf-fill.clean { background: linear-gradient(90deg, #4ADE80, #22C55E); }
  .prob-row { display: flex; justify-content: space-between; margin-top: 8px; font-size: .72rem; }
 
  .signals { padding: 1.25rem; border-top: 1px solid var(--border); }
  .signals-title {
    font-size: .68rem; font-weight: 700; color: var(--orange);
    text-transform: uppercase; letter-spacing: .12em; margin-bottom: .875rem;
  }
  .signal-item {
    display: flex; gap: 8px; align-items: flex-start;
    padding: 8px 11px;
    background: var(--orange-bg);
    border-radius: 7px; border: 1px solid var(--orange-bd);
    font-size: .8rem; color: var(--text-dim); line-height: 1.5; margin-bottom: 6px;
    animation: fadeIn .25s ease both;
  }
  .signal-item.ok {
    background: var(--green-bg);
    border-color: var(--green-bd);
    color: var(--green);
  }
  @keyframes fadeIn { from{opacity:0;transform:translateX(8px)} to{opacity:1;transform:none} }
 
  /* ── SPINNER ── */
  .spinner { display: none; padding: 3rem; text-align: center; }
  .spinner.show { display: block; }
  .spin {
    width: 36px; height: 36px;
    border: 3px solid rgba(255,255,255,.1);
    border-top-color: var(--orange);
    border-radius: 50%;
    animation: spin 1s linear infinite; margin: 0 auto 1rem;
  }
  @keyframes spin { to{transform:rotate(360deg)} }
  .spin-text { font-size: .8rem; color: var(--text-muted); }
 
  /* ── DATABASE ── */
  .section-title { font-size: .95rem; font-weight: 700; margin-bottom: .875rem; color: var(--orange); }
  .search-row { display: flex; gap: .75rem; flex-wrap: wrap; margin-bottom: .875rem; align-items: center; }
  .search-box {
    flex: 1; min-width: 200px;
    display: flex; gap: 8px; align-items: center;
    background: var(--navy-card); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 12px;
    transition: .15s;
  }
  .search-box:focus-within { border-color: var(--orange-bd); }
  .search-box input {
    background: none; border: none; outline: none;
    color: var(--text); font-family: inherit; font-size: .875rem; width: 100%;
  }
  .search-box input::placeholder { color: var(--text-muted); }
 
  .filter-btn {
    padding: 7px 14px; border: 1px solid var(--border2); border-radius: 8px;
    background: transparent; color: var(--text-dim);
    font-size: .75rem; cursor: pointer; font-family: inherit;
    transition: .15s; font-weight: 600;
  }
  .filter-btn:hover { border-color: var(--orange-bd); color: var(--orange); }
  .filter-btn.f-all   { background: var(--orange); border-color: var(--orange); color: #1A2744; }
  .filter-btn.f-scam  { background: var(--red-bg); border-color: var(--red-bd); color: var(--red); }
  .filter-btn.f-clean { background: var(--green-bg); border-color: var(--green-bd); color: var(--green); }
 
  .tbl-header {
    display: grid; grid-template-columns: 88px 1fr 170px; gap: .75rem;
    padding: .75rem 1.25rem;
    font-size: .65rem; font-weight: 700; color: var(--orange);
    text-transform: uppercase; letter-spacing: .12em;
    background: rgba(0,0,0,.25); border-bottom: 1px solid var(--border);
  }
  .history-row {
    display: grid; grid-template-columns: 88px 1fr 170px; gap: .75rem;
    padding: .875rem 1.25rem; border-bottom: 1px solid var(--border);
    font-size: .8rem; transition: .15s;
  }
  .history-row:last-child { border-bottom: none; }
  .clickable-row { cursor: pointer; }
  .clickable-row:hover {
    background: var(--orange-bg);
    border-left: 3px solid var(--orange);
    padding-left: calc(1.25rem - 2px);
  }
  .clickable-row:active { opacity: .8; }
 
  .badge-scam {
    display: inline-block; padding: 3px 9px;
    background: var(--red-bg); border: 1px solid var(--red-bd);
    border-radius: 5px; color: var(--red); font-size: .68rem; font-weight: 700; white-space: nowrap;
  }
  .badge-clean {
    display: inline-block; padding: 3px 9px;
    background: var(--green-bg); border: 1px solid var(--green-bd);
    border-radius: 5px; color: var(--green); font-size: .68rem; font-weight: 700; white-space: nowrap;
  }
  .text-clip   { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-dim); }
  .reason-clip { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-muted); font-size: .75rem; }
 
  .pager {
    padding: .875rem 1.25rem;
    display: flex; justify-content: space-between; align-items: center;
    background: rgba(0,0,0,.2); border-top: 1px solid var(--border);
  }
  .pager-info { font-size: .72rem; color: var(--text-muted); }
  .pager-btns { display: flex; gap: 4px; }
  .pager-btn {
    padding: 5px 11px; border: 1px solid var(--border2); border-radius: 6px;
    background: transparent; color: var(--text-dim);
    font-size: .75rem; cursor: pointer; font-family: inherit; transition: .15s;
  }
  .pager-btn:hover:not(:disabled) { border-color: var(--orange-bd); color: var(--orange); }
  .pager-btn:disabled { opacity: .25; cursor: not-allowed; }
  .pager-btn.active { background: var(--orange); border-color: var(--orange); color: #1A2744; font-weight: 700; }
</style>
</head>
<body>
 
<header>
  <div>
    <div class="logo">ANTI<span>SCAM</span> Detector</div>
    <div class="logo-sub">Nền tảng cảnh báo lừa đảo · AI</div>
  </div>
  <div class="badge"><div class="dot"></div> AI đang hoạt động</div>
</header>
 
<main>
  <div class="grid">
 
    <!-- INPUT PANEL -->
    <div>
      <div class="card">
        <div class="card-header">
          <span> Nội dung cần kiểm tra</span>
          <span id="char-cnt" style="font-family:monospace;font-size:.7rem;color:var(--text-muted)">0 ký tự</span>
        </div>
        <textarea id="input-text"
          placeholder="Dán tin nhắn, email hoặc nội dung đáng ngờ vào đây..."
          oninput="updateChar()"></textarea>
        <div class="card-footer">
          <div class="chips">
            <span class="chip" onclick="loadEx(\'scam1\')"> Nhập học giả</span>
            <span class="chip" onclick="loadEx(\'scam2\')"> Giả công an</span>
            <span class="chip" onclick="loadEx(\'scam3\')"> Đầu tư lừa</span>
            <span class="chip" onclick="loadEx(\'clean1\')"> Thông báo thật</span>
          </div>
          <button class="btn" id="btn-analyze" onclick="analyze()"> Phân tích</button>
        </div>
      </div>
 
      <!-- DATABASE -->
      <div style="margin-top:1.5rem;">
        <div class="section-title"> Cơ sở dữ liệu mẫu</div>
        <div class="search-row">
          <div class="search-box">
            <span style="color:var(--text-muted);font-size:.875rem"></span>
            <input type="text" id="db-search" placeholder="Tìm kiếm nội dung hoặc lý do..." oninput="filterDB()">
          </div>
          <button class="filter-btn f-all"  id="fb-all"   onclick="setFilter(\'all\')">Tất cả</button>
          <button class="filter-btn"         id="fb-scam"  onclick="setFilter(\'scam\')"> SCAM</button>
          <button class="filter-btn"         id="fb-clean" onclick="setFilter(\'clean\')"> CLEAN</button>
        </div>
        <div class="card">
          <div class="tbl-header">
            <div>Nhãn</div><div>Nội dung</div><div>Lý do</div>
          </div>
          <div id="db-rows"></div>
          <div class="pager">
            <div class="pager-info" id="pager-info"></div>
            <div class="pager-btns" id="pager-btns"></div>
          </div>
        </div>
      </div>
    </div>
 
    <!-- RESULT PANEL -->
    <div class="card" style="position:sticky;top:16px;">
      <div class="card-header"><span> Kết quả phân tích</span></div>
 
      <div class="spinner" id="spinner">
        <div class="spin"></div>
        <div class="spin-text" id="spin-text">Đang phân tích...</div>
      </div>
 
      <div id="result-placeholder" class="result-placeholder">
        <div class="result-placeholder-icon"></div>
        Nhập nội dung và nhấn <strong>Phân tích</strong>
      </div>
 
      <div id="result-content">
        <div class="verdict" id="verdict">
          <span class="verdict-icon" id="v-icon"></span>
          <div class="verdict-label" id="v-label"></div>
          <div class="verdict-sub"   id="v-sub"></div>
        </div>
        <div class="conf-wrap">
          <div class="conf-row">
            <span>Độ tin cậy phân tích</span>
            <span id="v-conf" style="font-family:monospace;color:var(--orange);font-weight:700">0%</span>
          </div>
          <div class="conf-bar"><div class="conf-fill" id="conf-fill"></div></div>
          <div class="prob-row">
            <span style="color:var(--text-muted)">P(SCAM): <b id="v-scam-p"  style="color:var(--red)"></b></span>
            <span style="color:var(--text-muted)">P(CLEAN): <b id="v-clean-p" style="color:var(--green)"></b></span>
          </div>
        </div>
        <div class="signals">
          <div class="signals-title"> Dấu hiệu phát hiện</div>
          <div id="signal-list"></div>
        </div>
      </div>
    </div>
 
  </div>
</main>
 
<script>
const EXAMPLES = {
  scam1: \'Xin chúc mừng! Bạn đã chính thức trúng tuyển vào Trường Đại học Khoa học Tự nhiên khóa K70. Để hoàn tất thủ tục nhập học, bạn vui lòng truy cập bit.ly/nhaphoc2024 và điền đầy đủ thông tin cá nhân. Đồng thời, bạn cần nộp khoản phí xác nhận nhập học là 2.500.000 VNĐ vào tài khoản ngân hàng 1234567890. Việc này phải hoàn thành trước 17h00 hôm nay!\',
  scam2: \'CÔNG AN THÀNH PHỐ THÔNG BÁO KHẨN CẤP! Tài khoản của bạn liên quan đến đường dây rửa tiền xuyên quốc gia. Để tránh bị khởi tố hình sự, bạn cần chuyển ngay toàn bộ số tiền vào tài khoản giám sát an toàn của chúng tôi. Nếu không hợp tác trong vòng 2 giờ, tài sản sẽ bị phong tỏa. Tuyệt mật, không được tiết lộ cho bất kỳ ai.\',
  scam3: \'Cơ hội làm giàu không giới hạn! Nền tảng đầu tư cam kết lợi nhuận 30% mỗi tuần. Chỉ cần nạp tối thiểu 2 triệu đồng, bạn nhận hoa hồng ngay lập tức và rút tiền bất cứ lúc nào. Hàng nghìn người đã thay đổi cuộc sống. Tham gia ngay hôm nay!\',
  clean1: \'Các tân sinh viên K70 thân mến! Thông tin chính thức về thủ tục nhập học được đăng trên cổng thông tin: https://hus.vnu.edu.vn. Phòng Công tác sinh viên luôn sẵn sàng hỗ trợ tại cơ sở 334 Nguyễn Trãi. Tất cả câu lạc bộ chính thức đều KHÔNG thu bất kỳ lệ phí gia nhập nào.\'
};
 
const DB_DATA = {{ db_data | safe }};
let dbFilter = \'all\', dbSearch = \'\', dbPage = 1;
const PAGE_SIZE = 12;
 
function updateChar() {
  const n = document.getElementById(\'input-text\').value.length;
  document.getElementById(\'char-cnt\').textContent = n.toLocaleString() + \' ký tự\';
}
 
function loadEx(key) {
  document.getElementById(\'input-text\').value = EXAMPLES[key];
  updateChar();
}
 
async function analyze() {
  const text = document.getElementById(\'input-text\').value.trim();
  if (!text) return;
  document.getElementById(\'btn-analyze\').disabled = true;
  document.getElementById(\'result-placeholder\').style.display = \'none\';
  document.getElementById(\'result-content\').style.display = \'none\';
  document.getElementById(\'spinner\').classList.add(\'show\');
  const steps = [\'Làm sạch văn bản...\',\'Trích xuất đặc trưng...\',\'Vector hoá TF-IDF...\',\'Dự đoán...\'];
  let si = 0;
  const itv = setInterval(() => {
    document.getElementById(\'spin-text\').textContent = steps[si++ % steps.length];
  }, 500);
  try {
    const res  = await fetch(\'/predict\', {
      method: \'POST\', headers: {\'Content-Type\': \'application/json\'},
      body: JSON.stringify({text})
    });
    const data = await res.json();
    clearInterval(itv);
    document.getElementById(\'spinner\').classList.remove(\'show\');
    showResult(data);
  } catch(e) {
    clearInterval(itv);
    document.getElementById(\'spinner\').classList.remove(\'show\');
    alert(\'Lỗi kết nối server: \' + e.message);
  }
  document.getElementById(\'btn-analyze\').disabled = false;
}
 
function showResult(r) {
  const isScam = r.label === \'SCAM\';
  document.getElementById(\'verdict\').className = \'verdict \' + (isScam ? \'scam\' : \'clean\');
  document.getElementById(\'v-label\').textContent = isScam ? \'LỪA ĐẢO\' : \'AN TOÀN\';
  document.getElementById(\'v-sub\').textContent   = isScam
    ? \'Tin nhắn có dấu hiệu lừa đảo!\'
    : \'Không phát hiện dấu hiệu lừa đảo\';
  document.getElementById(\'v-conf\').textContent    = r.confidence + \'%\';
  document.getElementById(\'v-scam-p\').textContent  = r.scam_prob  + \'%\';
  document.getElementById(\'v-clean-p\').textContent = r.clean_prob + \'%\';
  const fill = document.getElementById(\'conf-fill\');
  fill.className = \'conf-fill \' + (isScam ? \'scam\' : \'clean\');
  setTimeout(() => { fill.style.width = r.confidence + \'%\'; }, 80);
  const sl = document.getElementById(\'signal-list\');
  sl.innerHTML = \'\';
  (r.signals || []).forEach((s, i) => {
    const d = document.createElement(\'div\');
    d.className = \'signal-item\' + (s.startsWith(\'✅\') ? \' ok\' : \'\');
    d.style.animationDelay = (i * 0.08) + \'s\';
    d.textContent = s;
    sl.appendChild(d);
  });
  document.getElementById(\'result-content\').style.display = \'block\';
}
 
function getFiltered() {
  return DB_DATA.filter(d => {
    const mf = dbFilter === \'all\' || d.Label === dbFilter.toUpperCase();
    const ms = !dbSearch
      || d.Sub_Content.toLowerCase().includes(dbSearch)
      || (d.Reason||\'\'). toLowerCase().includes(dbSearch);
    return mf && ms;
  });
}
 
function renderDB() {
  const filtered   = getFiltered();
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  dbPage = Math.min(dbPage, totalPages);
  const start = (dbPage - 1) * PAGE_SIZE;
  const items = filtered.slice(start, start + PAGE_SIZE);
  window._currentItems = items;
  document.getElementById(\'db-rows\').innerHTML = items.map((d, i) => `
    <div class="history-row clickable-row" onclick="loadFromDB(${i})"
         title="Click de kiem tra tin nhan nay">
      <div><span class="${d.Label===\'SCAM\'?\'badge-scam\':\'badge-clean\'}">${d.Label}</span></div>
      <div class="text-clip">${esc(d.Sub_Content)}</div>
      <div class="reason-clip">${esc(d.Reason||\'—\')}</div>
    </div>
  `).join(\'\');
  document.getElementById(\'pager-info\').textContent =
    `${start+1}–${Math.min(start+PAGE_SIZE, filtered.length)} / ${filtered.length}`;
  const pb = document.getElementById(\'pager-btns\');
  pb.innerHTML = \'\';
  const add = (lbl, pg, dis=false, act=false) => {
    const b = document.createElement(\'button\');
    b.className = \'pager-btn\' + (act ? \' active\' : \'\');
    b.textContent = lbl; b.disabled = dis;
    b.onclick = () => { dbPage = pg; renderDB(); };
    pb.appendChild(b);
  };
  add(\'←\', dbPage-1, dbPage<=1);
  const s = Math.max(1, dbPage-2), e = Math.min(totalPages, s+4);
  for (let p=s; p<=e; p++) add(p, p, false, p===dbPage);
  add(\'→\', dbPage+1, dbPage>=totalPages);
}
 
function setFilter(f) {
  dbFilter = f; dbPage = 1;
  [\'all\',\'scam\',\'clean\'].forEach(k => {
    document.getElementById(\'fb-\'+k).className =
      \'filter-btn\' + (k===f ? \' f-\'+f : \'\');
  });
  renderDB();
}
 
function filterDB() {
  dbSearch = document.getElementById(\'db-search\').value.toLowerCase();
  dbPage = 1; renderDB();
}
 
function loadFromDB(i) {
  const d = window._currentItems[i];
  if (!d) return;
  const ta = document.getElementById(\'input-text\');
  ta.value = d.Sub_Content;
  updateChar();
  ta.scrollIntoView({ behavior: \'smooth\', block: \'center\' });
  ta.focus();
  ta.classList.add(\'highlight-pulse\');
  setTimeout(() => ta.classList.remove(\'highlight-pulse\'), 800);
}
 
function esc(s) {
  return String(s).replace(/&/g,\'&amp;\').replace(/</g,\'&lt;\').replace(/>/g,\'&gt;\').replace(/"/g,\'&quot;\');
}
 
renderDB();
</script>
</body>
</html>
"""
 
 
# Flask routes
 
@app.route("/")
def index():
    """Trang chủ — trả HTML với dữ liệu mẫu nhúng sẵn."""
    import json
    records = df_data[["Sub_Content", "Label", "Reason"]].fillna("").to_dict(orient="records")
    db_json = json.dumps(records, ensure_ascii=False)
    return render_template_string(HTML_TEMPLATE, db_data=db_json)
 
 
@app.route("/predict", methods=["POST"])
def predict():
    """API dự đoán — nhận JSON {text: ...}, trả JSON kết quả."""
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "Không có nội dung"}), 400
    try:
        result = predict_text(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
 
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """API phân tích hàng loạt — nhận JSON {texts: [...]}."""
    data  = request.get_json()
    texts = (data or {}).get("texts", [])
    if not texts:
        return jsonify({"error": "Không có dữ liệu"}), 400
    results = [predict_text(t) for t in texts]
    return jsonify(results)
 
 
@app.route("/stats")
def stats():
    """Thống kê nhanh về dataset."""
    total = len(df_data)
    scam  = int((df_data["Label"] == "SCAM").sum())
    clean = int((df_data["Label"] == "CLEAN").sum())
    return jsonify({
        "total": total, "scam": scam, "clean": clean,
        "scam_pct": round(scam / total * 100, 1) if total else 0,
        "model": type(model).__name__,
    })
 
 
# ──────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print("Server is running...")
    app.run(host="0.0.0.0", port=port, debug=True, )
 
 