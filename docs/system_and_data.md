# §3 Hệ thống & Dữ liệu

## 3.1 Kiến trúc hai tầng

Hệ thống phân loại giao dịch theo ba hành động: **cho qua (PASS)**, **chuyển tầng 2 (ESCALATE)**, **chặn (BLOCK)**. Quyết định dựa trên quy tắc hai ngưỡng: với điểm tầng 1 là `s`,

- `s < t_L` → PASS  
- `t_L ≤ s < t_H` → ESCALATE  
- `s ≥ t_H` → BLOCK

**Ma trận chi phí** (mỗi giao dịch):

| | Thực là gian lận | Thực là hợp lệ |
|---|---|---|
| PASS | 100 (bỏ sót) | 0 |
| ESCALATE | 0,2 + 100·1[tầng 2 bỏ sót] | 0,2 + 1·1[tầng 2 chặn nhầm] |
| BLOCK | 0 | 1 (chặn nhầm) |

**Tầng 1 — Hồi quy logistic:**  
Fit trên tập con đặc trưng (Amount, V1–V10 — placeholder, nhóm cần quyết định). Điểm thô `p_raw` được hiệu chuẩn bằng **Platt scaling ngoài fold**: 5-fold CV trên tập train sinh điểm OOF; Platt fit trên `(p_raw_OOF, y_train)`. Mô hình cuối fit trên toàn bộ train; val và test dùng Platt của OOF. Dòng train dùng điểm OOF (không dùng điểm mô hình cuối) để tránh rò rỉ dữ liệu khi chọn ngưỡng và tính bảng chi phí.

**Tầng 2 — Gradient Boosting:**  
Fit trên tất cả đặc trưng (Amount, V1–V28). Ngưỡng `t2` cố định từ OOF 5-fold trên train (tối thiểu chi phí `100·miss + 1·false_block`). Dòng train dùng quyết định OOF; val/test dùng quyết định của mô hình cuối.

**Vai trò từng tập:**

| Tập | Dùng để |
|---|---|
| Train | Fit mô hình, Platt OOF, chọn `t2`, tính biên khoảng |
| Val | Chọn chính sách (ngưỡng t_L, t_H) qua bảng chi phí |
| Test | Báo cáo một lần duy nhất sau khi khóa chính sách |

---

## 3.2 Dữ liệu

**Nguồn:** Credit Card Fraud Detection, ULB Machine Learning Group  
OpenML ID: 1597 | Giấy phép: DbCL v1.0  
**Đặc trưng:** Time, Amount, V1–V28 (28 thành phần PCA ẩn danh)  
**Nhãn:** Class (0 = hợp lệ, 1 = gian lận)

**Phiên bản file (SHA-256):**
```
Raw CSV:   2f2ca793ed45d753b75896d72dd6c187c8520ea0127c438106d55f0238da47a8
Splits:    48152ae8e2649a79912c935272df64de362640a03c2bf16f67c3fd040413a2b8
Manifest:  data/manifest.json
```

**Thống kê từ manifest (v2.0, group-aware):**

| Chỉ số | Giá trị |
|---|---|
| Tổng dòng (raw) | 284.807 |
| Số gian lận | 492 (0,173%) |
| Số giao dịch hợp lệ | 284.315 |
| Số nhóm đặc trưng (n_groups) | 275.663 |
| Dòng thuộc nhóm trùng | 14.293 |
| Bản dư Σ(nₕ−1) (excess_rows) | 9.144 |
| Kích thước nhóm lớn nhất | 77 |
| Nhóm xung đột nhãn | 0 |
| Phân chia mỗi split (xấp xỉ 60/20/20) | Train ~170.800 / Val ~57.000 / Test ~57.000 |
| Phân bố gian lận mỗi split | Train ~295–301 / Val ~95–99 / Test ~96–101 |

---

## 3.3 Quy trình đóng băng dữ liệu

```
creditcard.csv (raw)
       │
       ▼
load_and_inspect_raw()
  ├─ kiểm NaN/Inf
  ├─ tính group_id = groupby(ALL_FEATURES).ngroup()
  └─ ghi n_groups, excess_rows, conflicting_groups → manifest
       │
       ▼
generate_splits()  [StratifiedGroupKFold(5), 20 seeds]
  ├─ seed s: fold(s%5)=test, fold((s+1)%5)=val, còn lại=train
  ├─ kiểm tra: không giao nhau, không rò nhóm
  └─ hash SHA-256 từng mảng index → manifest
       │
       ▼
splits_20.npz + manifest.json
       │
       ▼
verify: python scripts/freeze_creditcard.py --verify
  → INTEGRITY CHECK PASSED
```

Lệnh tái tạo từ đầu:
```bash
python scripts/freeze_creditcard.py        # sinh splits + manifest
python scripts/freeze_creditcard.py --verify  # xác minh tính toàn vẹn
```

---

## 3.4 Mô tả T2 — Kiểm thử Khớp Vét Cạn

File `tests/test_search_vs_bruteforce.py` kiểm chứng hai bộ tìm ngưỡng (`best_two_threshold` trên lưới khoảng, `exact_two_threshold` trên điểm liên tục) bằng cách đối chiếu với thuật toán vét cạn độc lập trên 44 ca kiểm thử.

**Kết quả: 44/44 passed (0,34 s)**

Bốn tính chất được xác nhận:

1. **Chính xác tối ưu:** risk của thuật toán khớp vét cạn trong tolerance 1e-11 (0 sai lệch trên mọi ca).
2. **Score trùng không bị tách:** các giao dịch cùng score nhận cùng hành động.
3. **Hòa chi phí:** khi nhiều cặp ngưỡng đồng tối ưu, risk vẫn bằng nhau (không phụ thuộc vào nghiệm cụ thể được chọn).
4. **Thứ bậc rủi ro:** $R_{\text{oracle}} \leq R_{\text{B2}} \leq R_{\text{A1}}$ được kiểm tra trên 60 điểm ngẫu nhiên.

