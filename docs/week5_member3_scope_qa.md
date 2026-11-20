# Tuần 5 — Thành viên 3: Scope Lock và QA khả năng tái lập

**Khoảng thời gian:** 16/11–22/11/2026  
**Commit được audit:** `8e87cc9eaacc8bdfaa9b8a3921432fabe84716e3` (`test_chow`)  
**Nguyên tắc:** không thêm mô hình, baseline, metric hoặc lưới mới; chỉ kiểm tra mã, test, dữ liệu đóng băng và artifact hiện có.

## 1. Phạm vi được khóa

### Câu hỏi và chi phí

- Dataset: Credit Card Fraud Detection, OpenML ID 1597.
- Ba hành động: `PASS`, `ESCALATE`, `BLOCK`.
- Chi phí cố định theo đặc tả hiện tại: `C_FN=100`, `C_FP=1`, `C_ESC=0.2`.
- So sánh chính: chính sách tối ưu theo bin với thiết kế hai ngưỡng trên cùng bảng chi phí; ROC baseline và exact two-threshold chỉ là đối chứng đã có.

### Lưới/chạy thực nghiệm đã có artifact

| Hạng mục | Giá trị khóa | Bằng chứng |
|---|---|---|
| Seeds | `0..19` | `data/splits_20.npz`, 60 arrays (train/val/test) |
| Tỷ lệ split | 60/20/20, stratified | `data/manifest.json` |
| `K` | 10 | `outputs/creditcard_results.json`, `creditcard_fraud_quantile.json` |
| Bin method | `logit`, `fraud_quantile` | hai artifact 20-split hiện có |
| `min_pos` | 5 | trường `args` trong artifact |
| Bootstrap | 500 | trường `args` trong artifact |
| Tầng 1 | Logistic Regression, `Amount,V1..V10` | `scripts/run_creditcard.py` |
| Tầng 2 | HistGradientBoosting, `Amount,V1..V28` | `scripts/run_creditcard.py` |

Không coi `joint_probe` (`n_splits=1`, `n_boot=100`) là kết quả chính. Smoke dùng lưới riêng `K=16`, `min_pos=5`, `n_boot=300`, chỉ là kiểm tra tích hợp trên dữ liệu tổng hợp.

## 2. Kết quả QA

| Trạng thái | Kiểm tra | Kết quả/bằng chứng |
|---|---|---|
| PASS | Unit/integration tests | `python -m pytest tests/ -q`: **120 passed** trong 23.23s; chỉ có cảnh báo không ghi được `.pytest_cache` do filesystem read-only. |
| PASS | Archive split đúng checksum | SHA-256 thực tế và manifest cùng là `78811ec06059270780b8469b9d54f3f046653b24b92407887b6a27592c79cde7`. |
| PASS | Cấu trúc split | Có 60 arrays; seed 0 và 19 đều có kích thước 165397/55133/55133, không trùng index, hợp 275663 dòng; hash từng array khớp manifest. |
| PASS | Artifact JSON đọc được | Tất cả JSON trong `outputs/` parse thành công; các file kết quả chính có đủ 20 split và ghi rõ args. |
| PASS | Phản ví dụ tổng hợp | `smoke_results.json`: blind spot có mean relative regret 9.468%, policy phân mảnh 3/3 seed; constant có mean relative regret 0.099%. |
| FAIL | Calibration không dùng validation để fit | `scripts/run_creditcard.py` vẫn fit Platt bằng `p_raw[va], y[va]`, trong khi val còn dùng chọn policy. Điều này trái checklist trong `docs/test_spec.md`; chưa thể gọi pipeline dữ liệu thật là leakage-safe. |
| FAIL | Gom bản trùng theo nhóm trước split | Code đang `drop_duplicates(features + label)` rồi split từng hàng. Đặc tả yêu cầu gom theo vector đặc trưng, không dùng nhãn trong khóa, và không để một nhóm xuyên split. Manifest chưa ghi phân bố nhóm theo split. |
| FAIL | Đặc trưng tầng 1 đã chốt | `TIER1_FEATURES` còn chú thích `placeholder`; README cũng nói nhóm cần quyết định. Vì vậy chưa đạt scope lock về cấu hình mô hình. |
| FAIL | Tái lập dữ liệu đầu-cuối trên máy hiện tại | Không có `data/creditcard.csv` hoặc `data/creditcard_clean.csv`; chỉ có manifest và splits. `freeze_creditcard.py --verify` không thể kiểm chứng raw/processed checksum tại đây. |
| WARN | Đường dẫn manifest đa nền tảng | Manifest lưu `data\\creditcard.csv`, `data\\splits_20.npz`; chuỗi Windows không trực tiếp là path POSIX. |
| WARN | Artifact và source chưa cùng provenance | `creditcard_joint_system_roc.json` có trường joint/system ROC nhưng source `scripts/run_creditcard.py` hiện tại không sinh các trường đó. Cần gắn commit SHA vào mỗi run thay vì suy từ tên file. |
| WARN | README lệch số lượng test | README ghi 23 unit test, trong khi suite hiện có 120 test. |
| WARN | Diễn giải regret dữ liệu thật | Artifact `logit` cho regret bin-vs-two-threshold bằng 0 ở cả 20 split; `fraud_quantile` có mean regret -1.891%. Regret test âm là có thể do chọn policy trên val, nhưng không được trình bày như bằng chứng two-threshold vượt oracle Bayes. |

## 3. Quyết định Scope Lock

### Được phép trong Tuần 5

1. Chạy lại đúng cấu hình đã khóa và lưu log đầy đủ.
2. Sửa lỗi tuân thủ đặc tả dữ liệu (calibration/split theo nhóm) nếu nhóm xem đây là sửa lỗi, không phải tính năng.
3. Chốt danh sách đặc trưng tầng 1 bằng một quyết định duy nhất; sau khi chốt không thử thêm subset.
4. Bổ sung provenance cho artifact: commit SHA, trạng thái working tree, phiên bản môi trường, checksum input/splits, lệnh chạy và exit code.
5. Đồng bộ README/test spec với trạng thái thật.

### Không được phép

- Thêm dataset, model family, baseline, metric hoặc cơ chế tối ưu mới.
- Mở rộng joint optimization hay system-ROC thành nhánh nghiên cứu mới.
- Tìm lưới mới sau khi xem test để cải thiện kết quả.
- Thay đổi chi phí 100/1/0.2 hoặc tập seeds mà không ghi đó là sensitivity analysis đã định trước.

## 4. Thứ tự xử lý ưu tiên trước khi nghiệm thu

1. **P0 — chống leakage:** sửa và test quy trình calibration/split theo đúng `docs/test_spec.md`; sau đó chạy lại toàn bộ artifact dữ liệu thật.
2. **P0 — chốt feature set:** xóa trạng thái `placeholder` bằng quyết định nhóm có ghi biên bản.
3. **P1 — provenance:** mỗi output phải tự chứa SHA source/data/splits và cấu hình; loại khỏi báo cáo chính các artifact không truy về được source tương ứng.
4. **P1 — dữ liệu tái lập:** đặt raw/clean file ở môi trường chạy được phép và chạy `freeze_creditcard.py --verify`; không commit dữ liệu thô nếu giấy phép/quy định repo không cho phép.
5. **P2 — tài liệu:** cập nhật số test và phân biệt rõ validation-optimal, test cost, empirical bin regret và Bayes regret.

## 5. Tiêu chí hoàn thành Tuần 5

- Test suite pass trên **clean commit**, log có SHA và exit code.
- Checksum raw/clean/splits được verify; không có group xuyên train/val/test.
- Calibrator, scaler và model chỉ học từ train/OOF-train theo đặc tả; val chỉ chọn policy, test chỉ đánh giá.
- Feature set, chi phí, bin methods, K, min-positive, seeds và bootstrap được ghi cố định trong một cấu hình duy nhất.
- Các bảng/biểu đồ báo cáo sinh lại từ artifact có provenance, không trộn kết quả từ phiên bản source khác nhau.

**Kết luận QA:** phần toán/test lõi và archive splits đạt kiểm tra; pipeline dữ liệu thật **chưa đủ điều kiện nghiệm thu** vì còn lỗi calibration leakage, chính sách duplicate/group split chưa đạt đặc tả và feature set tầng 1 chưa thật sự khóa.
