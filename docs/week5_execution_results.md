> Lưu trữ audit/thực nghiệm ngày 20/11 trên nền 8e87cc9. Không phải đánh giá main hiện tại; xem `week5_merge_20261127.md` cho trạng thái sau tích hợp fix của Phước.

# G5-1 — báo cáo thực thi tuần 5 (20/11/2026)

## Kết quả thực hiện

Đã đọc mã và các tài liệu tuần 5; tải đúng Credit Card Fraud OpenML 1597; đo lại pipeline cũ trên **2 cấu hình × 20 seed**; sửa các lỗi tuân thủ dữ liệu mà Scope Lock cho phép; chạy lại **2 cấu hình × 20 seed** trên mã cuối cùng trong repository. Tất cả lượt chính thức hoàn thành, exit code 0. Lượt chạy sơ bộ trên bản sao đã dừng khi hoàn thiện provenance, không dùng làm kết quả chính.

Không thêm dataset, họ mô hình, baseline, metric, tối ưu hay lưới tham số. Giữ Amount,V1..V10 cho tầng 1 và Amount,V1..V28 cho tầng 2, không có Time. Đây là quyết định giữ nguyên cấu hình có sẵn theo Scope Lock, không phải tìm subset sau khi nhìn test.

## Sửa lỗi theo đặc tả

- `run_creditcard.py`: Platt học trên score OOF-train; scaler nằm trong pipeline từng fold. Train sử dụng score OOF, validation/test chỉ được transform. Ngưỡng tầng 2 cũng dùng OOF-train.
- `data_split.py`: định danh nhóm bằng feature float64 chính xác, không làm tròn, không dùng nhãn trong khóa; giữ mọi dòng, từ chối NaN/Inf. Cả outer split và OOF giữ nhóm không xuyên tập/fold. Nhãn any-positive chỉ dùng phân tầng nhóm, không xác định danh tính nhóm.
- Archive cũ/incomplete bị từ chối nếu không bao phủ đủ hàng hoặc có nhóm xuyên tập; không âm thầm dùng archive khử trùng cho dữ liệu giữ dòng.
- `freeze_week5.py`: đóng băng archive mới, ghi số hàng/nhóm/bản dư và hash từng split. Không ghi đè archive lịch sử.
- Smoke và fixture ROC hiệu chuẩn trên train. README cập nhật cách chạy, feature set và số test.
- Artifact dữ liệu thật tự chứa hash source/CSV/split, SHA Git, trạng thái working tree, phiên bản thư viện, lệnh và cấu hình. Log ghi exit code và thời gian.

## Lưới khóa và kiểm thử

20 seed 0..19; 60/20/20 theo nhóm; chi phí 100/1/0,2; K=10; min_pos=5; bootstrap 500; logit và fraud_quantile. LogisticRegression và HistGradientBoosting giữ nguyên họ mô hình và tham số cũ. Chia theo nhóm làm tỷ lệ **dòng** xấp xỉ: {'train': (0.5990021312678411, 0.6006207712591334), 'val': (0.1994894788400566, 0.20092202789959515), 'test': (0.19909623007861463, 0.20055686833539907)}.

- **124 tests pass**, gồm 4 test mới kiểm nhóm trùng/nhãn xung đột, archive lỗi, định danh float64 và bất biến dữ liệu hiệu chuẩn khi đổi nhãn val/test.
- **40/40 smoke checks pass**, exit code 0.
- Synthetic sau sửa: constant regret TB +0.099%; blind_spot +9.468%, phân mảnh 3/3 seed.
- Mã gốc: 120 tests, 40 smoke checks pass (báo cáo cũ ghi 39 smoke checks là sai số đếm).

## Dữ liệu thật — kết quả chính sau sửa

| Bin | Cost hai ngưỡng | Cost bin | Regret gộp | TB regret từng split | Phân mảnh | K hiệu dụng |
|---|---:|---:|---:|---:|---:|---:|
| logit | 0.057260 | 0.057260 | +0.000% | +0.000% | 0/20 | 4–5 |
| fraud_quantile | 0.036226 | 0.036306 | -0.223% | -0.170% | 2/20 | 10–10 |

- logit: 0 split dương, 0 âm, 20 bằng 0.
- fraud_quantile: 1 split dương, 1 âm, 18 bằng 0.

**Kết luận dữ liệu thật:** chưa có bằng chứng hai ngưỡng thua trong cấu hình đã khóa; logit bằng 0 ở mọi split, fraud_quantile gần 0 và trung bình âm. Không diễn giải phân mảnh đơn thuần là ưu thế trên test.

Regret gộp = (mean cost hai ngưỡng − mean cost bin) / mean cost bin; TB regret từng split = mean của từng tỷ số. Không đồng nhất hai đại lượng. Regret âm trên test hợp lệ vì so hai policy học trên validation, không phải hai tối ưu Bayes trên cùng phân phối.

CI95 bootstrap theo cặp từng split có trong `fixed_final/splits.csv`, JSON và `fixed_final/regret_by_split.png`. Không diễn giải CI từng split thành CI cho trung bình 20 split. Các split dùng chung dữ liệu nên không tạo CI tổng hợp bằng giả định độc lập; CI tổng hợp cho câu hỏi nghiên cứu vẫn chưa có trong pipeline đã khóa.

Không kết luận nhân quả riêng cho thay đổi calibration từ phép so trước/sau: sửa cả calibration, giữ dòng và group split cùng lúc. Không chọn kết quả đẹp hơn; báo cáo chính dùng duy nhất protocol sau sửa. Kết quả không tự chứng minh định lý Bayes; phản ví dụ giải tích và synthetic là bằng chứng về cơ chế.

## Dữ liệu tải và provenance

Nguồn: https://data.openml.org/datasets/0000/1597/dataset_1597.pq (URL từ metadata OpenML 1597). Có 284.807 dòng, 492 gian lận. Protocol mới giữ đủ hàng; 275663 nhóm, 5149 nhóm trùng, 14293 hàng thuộc nhóm trùng, 9144 bản dư.

Bản CSV xuất từ parquet có SHA khác raw CSV lịch sử. Đã ghi checksum mới, không sửa manifest cũ hoặc tuyên bố byte identity với CSV cũ. Lượt tái chạy nguyên trạng cho cùng các số tổng hợp lịch sử trên cả hai bin methods; đối chiếu này không thay cho chứng minh byte identity. Chi tiết `download_data_audit.json`, `openml_1597.json` và `fixed_final/manifest.json`.

CSV lưu ở `data/creditcard_openml_1597.csv` trong repository. Archive mới: `data/splits_week5_grouped.npz`; manifest mới: `data/manifest_week5_grouped.json`. Mỗi split được kiểm bao phủ đủ mọi hàng đúng một lần và không xuyên nhóm. Hash source trong cả hai artifact cuối được đối chiếu lại với file thật sau khi chạy.

## Phân loại bằng chứng và việc còn lại

- `fixed_final/`: kết quả chính sau sửa, 40 lượt, cùng một phiên bản mã.
- `rerun_*.json`, `historical_*`, `BASELINE_TRUOC_SUA.md`: kết quả trước sửa/exploratory, không trộn với bảng chính.
- `tests.log`, `smoke.log`: test mã gốc; `fixed_final/tests.log`, `fixed_final/smoke.log`: test mã cuối.
- Không thay định lý hay mở rộng nhánh joint/system ROC.

Đã hoàn thành phần triển khai và đo đạc cục bộ tuần 5. **Chưa gọi là nghiệm thu trên clean commit/CI**: thay đổi chưa commit, SHA nền là `8e87cc9eaacc8bdfaa9b8a3921432fabe84716e3` và mọi file chạy được nhận diện bằng SHA-256. Cần nhóm review/commit và chạy CI theo quy trình nộp. Không tự tạo commit hay ghi nhận phê duyệt nhóm thay người dùng. CI tổng hợp cho trung bình 20 split vẫn là giới hạn báo cáo nêu trên.

## Chạy lại

Xem README đã cập nhật. Dùng `scripts/freeze_week5.py --csv data/creditcard_openml_1597.csv` để tái tạo archive; chạy `scripts/run_creditcard.py` riêng cho logit và fraud_quantile, với `--n-splits 20 --K 10 --min-pos 5 --n-boot 500 --splits data/splits_week5_grouped.npz`. Ghi `--out` riêng theo method để không ghi đè kết quả.
