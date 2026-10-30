# Khi hành động trì hoãn là một bộ phân lớp

**Chow's Rule Fails When the Abstention Is a Classifier: Exact Conditions for Interval-Optimality of Three-Action Risk Policies, and the Regret of Two-Threshold Designs**

Đồ án tổng hợp CO3101 · Trường ĐH Bách Khoa, ĐHQG-HCM · Khoa KH&KT Máy tính
Nhóm: `[tên nhóm]` · `[Họ tên – MSSV]` · GVHD: `[tên]`

## Câu hỏi nghiên cứu

Trên Credit Card Fraud (tầng 1 hồi quy logistic đã hiệu chuẩn, tầng 2 gradient boosting, chi phí bỏ sót / chặn nhầm / leo thang = 100 / 1 / 0,2), chi phí kiểm thử trên mỗi giao dịch của **chính sách hai ngưỡng tốt nhất** cao hơn **chính sách ba hành động tối ưu theo từng khoảng điểm** bao nhiêu phần trăm, trung bình qua 20 lần chia dữ liệu (kèm khoảng tin cậy 95%)?

## Chạy nhanh (smoke test, không cần mạng, ~10 giây)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make smoke            # hoặc: python scripts/smoke_test.py
make test             # 23 unit test (~20 giây)
```

Kết quả mong đợi: dòng cuối là `SMOKE TEST PASSED`, mã thoát 0, và file `outputs/smoke_results.json`.

Smoke test chạy toàn bộ chuỗi *sinh dữ liệu → chia 60/20/20 → hiệu chuẩn Platt → chia khoảng → bảng chi phí → hai chính sách → chi phí và regret trên test* cho hai kịch bản tổng hợp đã biết trước đáp án:

| Kịch bản | Sai số tầng 2 | Điều phải thấy |
|---|---|---|
| `constant` | không đổi theo điểm | regret xấp xỉ 0, chính sách theo khoảng có dạng hai ngưỡng |
| `blind_spot` | tầng 2 "mù" trong một cửa sổ điểm | chính sách tối ưu bị đứt (leo thang → chặn → leo thang → chặn), regret > 0, ROC tốn hơn quét lưới rõ rệt |

Ngoài ra `perfect` (tầng 2 không sai, đúng trường hợp Chow: ngưỡng 0,002 và 0,8) dùng trong unit test. `synthetic.analytic_costs` tính đáp án giải tích bằng tích phân: ở `blind_spot`, hai ngưỡng tốt nhất tốn hơn chính sách tối ưu 9,07% và ROC tốn hơn 62,79%.

> Smoke test chỉ chứng minh **code nối đúng**, không phải kết quả khoa học. Dữ liệu tổng hợp do chúng tôi thiết kế để hiện tượng xuất hiện.

## Chạy trên dữ liệu thật

Dữ liệu **không** được commit (xem `.gitignore`). Tải Credit Card Fraud Detection (OpenML id 1597, hoặc bản Kaggle) và kiểm tra giấy phép trước khi dùng.

```bash
python scripts/run_creditcard.py --csv data/creditcard.csv --n-splits 20 --K 10 --method logit
# hoặc: python scripts/run_creditcard.py --openml --n-splits 20
```

Lệnh trên ghi `outputs/creditcard_results.json` và hình
`outputs/creditcard_results.png` (chi phí test của bốn chính sách qua từng lần
chia dữ liệu, cùng regret tương đối và khoảng tin cậy 95% theo bootstrap ghép
cặp trong từng tập test). Có thể vẽ lại từ JSON bằng
`python scripts/plot_creditcard.py outputs/creditcard_results.json`.
Các lần chia dùng lại nhiều giao dịch, vì vậy khoảng tin cậy của từng lần chia
không phải khoảng tin cậy cho trung bình qua 20 lần chia.

Chạy kiểm tra trên OpenML 1597 (loại trùng, 275.663 giao dịch / 473 gian lận,
20 lần chia, seed 0–19, chi phí 100/1/0,2, mặc định `min_pos=5`):

| Cách chia khoảng | K thực tế | Chi phí hai ngưỡng theo khoảng | Chi phí theo khoảng | Chênh lệch tương đối |
|---|---:|---:|---:|---:|
| `logit` (mặc định) | 3–5 | 0,06014 | 0,06014 | 0,00% |
| `fraud_quantile` | 10 | 0,03753 | 0,03825 | −1,89% |

`fraud_quantile` chạy với lệnh sau:

```bash
python scripts/run_creditcard.py --openml --n-splits 20 --method fraud_quantile --out outputs/creditcard_fraud_quantile.json
```

Chênh lệch âm trên test cho thấy
chính sách theo khoảng có thể khớp nhiễu validation; đây chưa phải bằng chứng
rằng hai ngưỡng tối ưu trong tổng thể. Các kết quả phụ thuộc cách chia khoảng,
tập đặc trưng tầng 1 tạm thời và giả định chi phí.

Kiểm tra sau khi tải: 284.807 dòng, 492 gian lận. Mặc định script loại các dòng trùng lặp và in số dòng trước/sau.
Bản OpenML 1597 không có cột `Time`; script dùng các đặc trưng còn lại và in
số đặc trưng được dùng. CSV Kaggle có `Time` thì dùng thêm cột này.

## Phương pháp tóm tắt

- **Ba hành động:** cho qua / leo thang / chặn. Chi phí: bỏ sót gian lận 100, chặn nhầm 1, leo thang 0,2 cộng lỗi của tầng 2 (`src/threeaction/costs.py`).
- **Tầng 1** cho điểm thô, **hiệu chuẩn Platt** học trên tập kiểm định thành điểm `s`. **Tầng 2** cho quyết định cuối với ngưỡng `t2` cố định một lần từ dự đoán ngoài mẫu trên train.
- **Chia khoảng:** biên khoảng tính từ **train**, dùng nguyên cho val/test (`bins.py`: `quantile`, `logit`, `fraud_quantile`, kèm gộp khoảng để mỗi khoảng đủ gian lận).
- **Baseline ROC-optimal** (`roc_baseline.py`, Pietraszek, ICML 2005): giả định chi phí leo thang là hằng số theo lớp (a⁺ cho gian lận, a⁻ cho giao dịch hợp lệ, ước lượng từ **train**), đường ROC lấy từ **val**. Đại diện cho thiết kế hai ngưỡng truyền thống. Khác quét lưới ở chỗ nó tối ưu một *mô hình* chi phí (hằng số), còn quét lưới (`exact_two_threshold`, O(n)) tối ưu chi phí thật. Hai cách trùng nhau khi chi phí leo thang đúng là hằng số.
- **Hai chính sách chọn từ cùng một bảng chi phí trên val** (`policies.py`): theo khoảng (mỗi khoảng chọn hành động rẻ nhất) và hai ngưỡng (cho qua → leo thang → chặn liền nhau, ngưỡng đặt trên biên khoảng nên là tập con của chính sách theo khoảng).
- **Đánh giá chỉ trên test:** chi phí trung bình mỗi giao dịch, regret tuyệt đối và tương đối, khoảng tin cậy bootstrap theo cặp.

## Giả định và hạn chế (ghi trước để không bị hiểu nhầm)

- Regret đo ở đây là so với chính sách tối ưu trong lớp "hàm của điểm tầng 1 chia K khoảng", ước lượng từ dữ liệu. **Không phải regret Bayes thật.**
- Tập kiểm định chỉ có khoảng 98 gian lận: chính sách theo khoảng có thể khớp nhiễu (đã quan sát trong smoke test, kịch bản `constant`, seed 3). Vì vậy K nhỏ, có gộp khoảng, và báo cáo kèm khoảng tin cậy; regret âm hoặc gần 0 là kết quả hợp lệ.
- Chia ngẫu nhiên, không xét trôi dữ liệu theo thời gian. 20 lần chia dùng chung dữ liệu nên không độc lập.
- Chi phí 100/1/0,2 là giả định, cần quét độ nhạy.
- Tập đặc trưng tầng 1 trong `run_creditcard.py` (Time, Amount, V1–V10) là **giá trị tạm**, nhóm cần quyết định.

## Cấu trúc thư mục

```
src/threeaction/   costs, bins, calibrate, tier2, policies, roc_baseline, pipeline, synthetic
scripts/           smoke_test.py (không cần mạng), run_creditcard.py (dữ liệu thật)
tests/             test_core.py, test_roc.py (23 test; unittest, chạy được với pytest)
data/  outputs/    dữ liệu thô và kết quả (không commit)
```

## Kế hoạch

- [ ] W4 hình thức hóa mô hình · [ ] W5 điều kiện lồi + phản ví dụ · [ ] W6 cận trên regret
- [x] W7 simulator tổng hợp + kiểm thử 1, 2 (khung đã có, cần đối chiếu với cận lý thuyết)
- [ ] W8 dữ liệu thật + kiểm tra sớm có đứt đoạn · [ ] W9 baseline · [ ] W10 chạy 20 lần chia
- [ ] W11 so cận với regret thực · [ ] W12 quét chi phí, độ nhạy K · [ ] W13–14 báo cáo, slide

## Tài liệu tham khảo

- Dal Pozzolo et al., Credit Card Fraud Detection dataset `[kiểm tra trích dẫn đầy đủ]`
- Narasimhan et al., NeurIPS 2022 `[kiểm tra trích dẫn đầy đủ]`
- Chow, C. K. (1970), *On optimum recognition error and reject tradeoff* `[kiểm tra]`

## Giấy phép

`[chọn giấy phép cho mã nguồn, ví dụ MIT]`. Dữ liệu tuân theo giấy phép của nguồn gốc.
