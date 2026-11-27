> Lưu trữ audit/thực nghiệm ngày 20/11 trên nền 8e87cc9. Không phải đánh giá main hiện tại; xem `week5_merge_20261127.md` cho trạng thái sau tích hợp fix của Phước.

# Tuần 5 — Thành viên 2: Kiểm chứng thực nghiệm

## Phạm vi đã khóa

Không thêm mô hình, tính năng hay lưới tham số. Chỉ kiểm tra test, smoke test và các artifact Credit Card Fraud đã có với cấu hình chính: 20 split cố định, tỉ lệ 60/20/20, chi phí bỏ sót/chặn nhầm/leo thang = 100/1/0,2, `K=10`, `min_pos=5`, bootstrap 500 lần.

## Kết quả kiểm chứng

- `python3 -m pytest tests/ -v`: **120/120 test pass** trong 23,49 giây. Có một cảnh báo do sandbox không ghi được `.pytest_cache`, không phải lỗi logic.
- `python3 scripts/smoke_test.py`: toàn bộ 39 kiểm tra nội dung đều hiển thị `[PASS]`. Kịch bản `constant` có regret trung bình `+0,10%`; `blind_spot` có regret trung bình `+9,47%` và chính sách theo khoảng bị phân mảnh ở 3/3 seed. Tiến trình trả mã 1 chỉ vì không thể ghi đè `outputs/smoke_results.json` trong sandbox chỉ-đọc.

## Kết quả dữ liệu Credit Card Fraud đã đóng băng

Manifest ghi nhận 284.807 dòng thô, sau khử trùng còn 275.663 dòng với 473 gian lận. Archive có 20 split phân tầng 60/20/20.

| Artifact | Chia khoảng | K hiệu dụng | Cost hai ngưỡng | Cost theo khoảng | Regret tương đối gộp | Phân mảnh |
|---|---:|---:|---:|---:|---:|---:|
| `creditcard_results.json` | logit | 3–5 (TB 4,25) | 0,060145 | 0,060145 | 0,00% | 0/20 split |
| `creditcard_fraud_quantile.json` | fraud-quantile | 10 | 0,037527 | 0,038250 | -1,89% | 6/20 split |

Với `fraud_quantile`, regret âm trên test không vi phạm tính tối ưu trên validation: hai chính sách được chọn trên validation rồi đánh giá trên test, nên chính sách linh hoạt hơn có thể overfit. Trong 20 split có 5 regret âm, 1 dương và 14 bằng 0; do đó **chưa có bằng chứng thực nghiệm rằng hai ngưỡng thua trên bộ dữ liệu thật này**.

Baseline ROC trong artifact logit có cost trung bình 0,036691 và exact two-threshold liên tục 0,034407, thấp hơn baseline theo bin. Các đại lượng này không cùng lớp chính sách rời rạc theo bin, nên không được diễn giải như một regret Bayes trực tiếp.

## Rủi ro và quyết định cho Tuần 5

1. README ghi “23 unit test” nhưng suite hiện tại thu thập 120 test; cần sửa con số trong báo cáo cuối, không cần đổi code thực nghiệm.
2. Kết quả `logit` bị gộp còn K=3–5 vì `min_pos=5`, làm mất độ phân giải; `fraud_quantile` giữ đủ K=10 nhưng không cho regret dương trên test.
3. Không chạy thêm lưới sau scope lock. Báo cáo trung thực: hiện tượng được xác nhận trên synthetic blind spot, nhưng chưa được xác nhận trên Credit Card Fraud với cấu hình đã chốt.
