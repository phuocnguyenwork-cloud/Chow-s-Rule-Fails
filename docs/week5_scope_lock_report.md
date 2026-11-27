> Lưu trữ audit/thực nghiệm ngày 20/11 trên nền 8e87cc9. Không phải đánh giá main hiện tại; xem `week5_merge_20261127.md` cho trạng thái sau tích hợp fix của Phước.

# TUẦN 5 (16/11–22/11/2026) — SCOPE LOCK

**Đề tài G5-1:** *Chow's Rule Fails When the Abstention Is a Classifier*  
**Commit được kiểm tra:** `8e87cc9eaacc8bdfaa9b8a3921432fabe84716e3`  
**Nguyên tắc:** dừng bổ sung tính năng; chỉ khóa giả thuyết, kiểm thử, đo trên lưới đã chốt và rà soát khả năng tái lập.

## 1. Phân công ba thành viên

| Thành viên | Phần việc Tuần 5 | Sản phẩm |
|---|---|---|
| 1 — Lý thuyết | Khóa mô hình risk ba hành động, điều kiện interval-optimality, cận regret và phản ví dụ phân mảnh | `docs/week5_member1_theory.md` |
| 2 — Thực nghiệm | Chạy test/smoke, tổng hợp các artifact Credit Card trên lưới đã khóa, kiểm tra regret và phân mảnh | `docs/week5_member2_experiments.md` |
| 3 — Scope & QA | Khóa cấu hình, kiểm tra split/checksum/provenance, lập checklist pass/fail và blocker nghiệm thu | `docs/week5_member3_scope_qa.md` |

## 2. Phạm vi đã khóa

- Dữ liệu thật: Credit Card Fraud Detection, OpenML ID 1597.
- Ba hành động: `PASS`, `ESCALATE`, `BLOCK`.
- Chi phí: bỏ sót `100`, chặn nhầm `1`, leo thang `0,2`.
- 20 seed cố định `0..19`; split phân tầng 60/20/20.
- `K=10`, `min_pos=5`, bootstrap 500.
- Hai cách chia bin đã có artifact chính: `logit` và `fraud_quantile`.
- Tầng 1 hiện tại: Logistic Regression trên `Amount,V1..V10`; tầng 2: HistGradientBoosting trên `Amount,V1..V28`.
- Không thêm dataset, họ mô hình, baseline, metric, chi phí hoặc lưới mới trong tuần này.

`joint_probe` chỉ có một split và 100 bootstrap nên không được dùng làm kết quả chính. Các artifact joint/system ROC chưa truy được về source hiện tại cũng không được dùng để đưa ra kết luận chính.

## 3. Kết quả lý thuyết đã khóa

Risk điều kiện của ba hành động:

\[
r_P(s)=As,\qquad r_B(s)=D(1-s),\qquad
r_E(s)=c+As\,m(s)+D(1-s)f(s).
\]

Trong trường hợp Chow cổ điển, khi chi phí leo thang là hằng số:

\[
t_L=c/A,\qquad t_H=1-c/D,
\]

nếu \(c<AD/(A+D)\). Nếu không, vùng leo thang rỗng và nghiệm suy biến tại \(t_L=t_H=D/(A+D)\). Với cấu hình dự án, hai ngưỡng giải tích là `0,002` và `0,8`.

Với risk leo thang tổng quát \(e(s)\), điều kiện chính xác được phát biểu theo bao dưới của ba risk: phải tồn tại \(t_L\le t_H\) sao cho PASS tối ưu ở bên trái, ESCALATE tối ưu ở giữa và BLOCK tối ưu ở bên phải, gần khắp nơi theo phân phối score. Trên lưới bin, điều này tương đương với việc tồn tại một chuỗi nghiệm `PASS* ESCALATE* BLOCK*`.

Regret do hạn chế hai ngưỡng thỏa:

\[
0\le \Delta=R_{2T}^*-R_S^*\le R_{A1}-R_S^*.
\]

Nếu dùng surrogate affine \(e_0\) có nghiệm Bayes hai ngưỡng, còn \(e\) là risk thật, thì:

\[
0\le\Delta\le \mathbb E|e(S)-e_0(S)|.
\]

Phản ví dụ 5 bin đã chốt cho chuỗi `PASS, ESCALATE, BLOCK, ESCALATE, BLOCK`, với \(R_S^*=0,18576\), \(R_{2T}^*=0,19149\) và \(\Delta=0,00573>0\).

**Giới hạn phát biểu:** ROC tầng 1 riêng lẻ không xác định sai số tầng 2 theo score, nên chưa đủ để suy ra interval-optimality. Không tuyên bố một định lý “ROC lồi/đơn điệu là đủ” nếu chưa bổ sung giả thiết và chứng minh.

## 4. Kết quả kiểm thử và thực nghiệm

### 4.1. Kiểm thử

- `python -m pytest tests/ -q`: **120 passed**; chỉ có cảnh báo môi trường không ghi được `.pytest_cache`.
- Bộ test lý thuyết mục tiêu: **53 passed**.
- Smoke test hiển thị **39/39 check PASS**. Tiến trình chỉ thất bại ở bước ghi đè JSON do filesystem của phiên làm việc ở chế độ chỉ đọc, không phải do điều kiện kiểm thử.

### 4.2. Dữ liệu tổng hợp

| Kịch bản | Regret tương đối trung bình | Chính sách phân mảnh |
|---|---:|---:|
| Chi phí tầng 2 hằng | +0,10% | 1/4 seed (do nhiễu học chính sách) |
| Blind spot phụ thuộc score | +9,47% | 3/3 seed |

Kết quả tổng hợp xác nhận cơ chế nghiên cứu: khi tầng 2 có vùng mù phụ thuộc score, chính sách tối ưu có thể bị đứt đoạn và thiết kế hai ngưỡng chịu regret dương.

### 4.3. Credit Card Fraud — artifact hiện có

| Chia bin | K hiệu dụng | Cost hai ngưỡng | Cost theo bin | Regret tương đối gộp | Phân mảnh |
|---|---:|---:|---:|---:|---:|
| `logit` | 3–5 | 0,060145 | 0,060145 | 0,00% | 0/20 |
| `fraud_quantile` | 10 | 0,037527 | 0,038250 | -1,89% | 6/20 |

Regret âm trên test không mâu thuẫn với tối ưu lý thuyết: chính sách được chọn trên validation và đánh giá trên test, nên lớp chính sách linh hoạt hơn có thể overfit. Với `fraud_quantile`, 14 split có regret bằng 0, 5 âm và 1 dương.

**Kết luận thực nghiệm trung thực:** hiện tượng đã được xác nhận trên dữ liệu tổng hợp có kiểm soát, nhưng **chưa được xác nhận trên Credit Card Fraud** với lưới đã khóa.

## 5. Trạng thái nghiệm thu và blocker

Phần toán, solver, test lõi và archive split đạt QA. Tuy nhiên pipeline dữ liệu thật **chưa đủ điều kiện nghiệm thu/kết luận khoa học** vì ba blocker P0:

1. `PlattCalibrator` đang fit trên validation, trong khi validation còn được dùng để chọn policy; điều này trái đặc tả chống leakage.
2. Quy trình hiện tại loại dòng trùng theo cả feature và label rồi split từng hàng; chưa bảo đảm các vector feature giống nhau không đi xuyên train/validation/test.
3. `TIER1_FEATURES` vẫn được ghi là `placeholder`, nên cấu hình mô hình chưa thật sự được khóa bằng quyết định nhóm.

Ngoài ra, máy kiểm tra không có raw/clean CSV để verify toàn tuyến; một số artifact system/joint ROC thiếu provenance khớp với source hiện tại; README ghi 23 test trong khi suite hiện có 120.

## 6. Quyết định cuối Tuần 5

- Giữ nguyên câu hỏi nghiên cứu, chi phí, seeds, mô hình, bin methods và lưới đã nêu.
- Không chạy thêm lưới nhằm tìm kết quả đẹp hơn.
- Chỉ cho phép sửa lỗi tuân thủ đặc tả: chống leakage, group split, chốt feature set và gắn provenance.
- Sau khi sửa các blocker, phải chạy lại đúng lưới đã khóa; artifact cũ chỉ được gắn nhãn **exploratory/không leakage-safe**.
- Báo cáo giữa kỳ dùng phản ví dụ tổng hợp làm bằng chứng dương; kết quả Credit Card được báo là null/inconclusive, không diễn giải regret âm như phủ định định lý.

## 7. Tiêu chí đóng Tuần 5

- Test pass trên clean commit, log có SHA và exit code.
- Raw/clean/splits được verify checksum; không có group feature xuyên các tập.
- Model, scaler và calibrator chỉ học từ train/OOF-train; validation chỉ chọn policy; test chỉ đánh giá.
- Feature set tầng 1 được chốt một lần bằng biên bản.
- Mỗi artifact ghi source SHA, trạng thái working tree, checksum input/splits, cấu hình và phiên bản môi trường.

Nếu chưa đạt đủ các mục trên, trạng thái đúng là **“đã scope lock và hoàn thành audit, chưa nghiệm thu thực nghiệm dữ liệu thật”**.
