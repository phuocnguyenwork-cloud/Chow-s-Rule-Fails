# Đặc tả nghiệm Chow và kiểm thử T1

**Phạm vi:** mã nguồn nhánh `modified-baseline`, gốc
`a0dce4698c666edbb1165b1174dc1ea3b0d63055`.
Commit sửa nguồn và bổ sung T1: `02da4c207f1e6dbb5f7edca6de3bc85f5779a637`.
Trạng thái chạy phải đọc cùng `validation_report.md` và các log pytest trong
thư mục `validation/` của bộ bàn giao. Tài liệu này không tự ghi pass.

## 1. Score, hành động và ma trận chi phí

- Y=1: gian lận/nguy hiểm; Y=0: hợp lệ. S tăng theo nguy cơ lớp 1.
- Khi suy luận công thức: P(Y=1 | S=s)=s. Nếu chưa hiệu chuẩn, dùng
  eta(s)=P(Y=1 | S=s), không thay score thô trực tiếp vào công thức xác suất.
- Cột action: PASS=0, ESCALATE=1, BLOCK=2. “Reject” của chính sách là BLOCK;
  không dùng cùng từ này cho abstain.
- A=C_FN>0, D=C_FP>0, phí gọi tầng 2 c>=0; cùng đơn vị chi phí/giao dịch.
  Quyết định đúng có chi phí 0.

| Nhãn | PASS | ESCALATE hoàn hảo | BLOCK |
|---|---:|---:|---:|
| Y=0 | 0 | c | D |
| Y=1 | A | c | 0 |

Tầng 2 bất toàn, quyết định b=1 là BLOCK:

    loss_E(y,b) = c + A*y*(1-b) + D*(1-y)*b
    r_P(s) = A*s
    r_B(s) = D*(1-s)
    r_E(s) = c + A*s*m(s) + D*(1-s)*f(s)

Ở đây m(s)=P(b=0|Y=1,S=s), f(s)=P(b=1|Y=0,S=s).

**Vì sao:** Chiều score và thứ tự action giữ đúng nghĩa của `costs.py`.
Chi phí bất đối xứng giúp phát hiện đảo nhãn và giả định sai về ngưỡng đối
xứng. Phí gọi vẫn phải trả khi tầng 2 sai, nên loss E phải cộng lỗi cuối.
Hiệu chuẩn cần cho kỳ vọng xác suất; ROC chỉ cần thứ hạng để tối ưu chi phí
thực nghiệm. Các số 100/1/0.2 là giả định thí nghiệm, cần quét độ nhạy,
không phải tham số được chứng minh từ dữ liệu.

## 2. Nghiệm Chow và điều kiện vùng rỗng

Khi r_E=c, giải giao điểm P/E và E/B:

    t_L=c/A; t_H=1-c/D
    t0=D/(A+D); c_crit=A*D/(A+D)

- c<c_crit: hai ngưỡng t_L<t_H; A=100,D=1,c=.2 cho .002 và .8.
- c>=c_crit: chọn t_L=t_H=t0, vùng ESC rỗng.
- Tại dấu bằng, E hòa ở riêng t0; quy ước chọn BLOCK ở đó vẫn Bayes tối ưu.

**Vì sao:** Phải chọn đường thấp nhất trong cả ba risk, không lấy giao điểm
máy móc. Điều kiện t_L<t_H chính là c<c_crit. Khi phí vượt mức này, E không
có khoảng rẻ hơn cả P lẫn B. Hoán đổi hoặc clip hai giao điểm không sửa được
vấn đề tối ưu. A lớn làm ngưỡng PASS thấp vì bỏ sót đắt hơn gọi tầng 2.

Chi phí hằng theo lớp a_plus,a_minus>=0:

    e0(s)=a_plus*s+a_minus*(1-s)
    t_L=a_minus/(A-a_plus+a_minus)
    t_H=(D-a_minus)/(D-a_minus+a_plus)

Vùng E có độ dài dương iff a_plus/A+a_minus/D<1; nếu không, chọn t0,t0.
**Vì sao:** So e0(t0) với risk P/B tại t0 cho đúng bất đẳng thức này; ba risk
affine tạo bao dưới thứ tự P,E,B. a_plus=a_minus=c là trường hợp Chow.

Nếu tầng 2 phụ thuộc score, xét tất cả vùng mà P/E/B tối thiểu. Không suy
ra hai ngưỡng tối ưu chỉ từ ROC tầng 1: ROC không xác định m(s),f(s).
Điều kiện cần và đủ tổng quát chưa được chứng minh bởi T1.

## 3. Hòa, vùng rỗng và tính nhất quán

| Điều kiện | Action |
|---|---|
| s<t_L | PASS |
| t_L<=s<t_H | ESCALATE |
| s>=t_H | BLOCK |
| t_L=t_H=t | PASS khi s<t; BLOCK khi s>=t; E rỗng |
| (-inf,-inf) | Toàn BLOCK |
| (+inf,+inf) | Toàn PASS |
| (-inf,+inf) | Toàn E, với score hữu hạn |

**Vì sao:** Gán biên sang phải khớp hàm ngưỡng và `assign_bins`. Đây là quy
ước tái lập, không phải yêu cầu toán học chọn action mạnh hơn tại hòa.
Cho phép vùng E rỗng giữ chính sách một ngưỡng A1 trong lớp hai ngưỡng.
Dùng +/-inf biểu diễn nghiệm một action mà không tạo nhánh dự đoán riêng.
Với K bin, cho phép 0<=i<=j<=K. Score bằng nhau không được tách action:
chính sách chỉ dùng S không có thông tin để phân biệt các hàng đó.

`per_bin_actions` dùng `np.argmin`, hòa ưu tiên P,E,B. Nó khác hàm ngưỡng ở
biên. Khi đối chiếu nghiệm Bayes ở hòa phải kiểm cùng risk, không ép action
bằng nhau. `best_two_threshold` dùng cải thiện tổng cost >1e-12, trong khi
exact solver dùng so sánh strict. Không gọi đây là một ưu tiên hòa thống
nhất toàn hệ thống, không bảo đảm cùng cặp ngưỡng khi nhiều nghiệm đồng hạng.

## 4. T1, oracle độc lập và tolerance

File: `tests/test_chow_constant.py`; chạy bằng pytest, kể cả các lớp kế thừa
`unittest.TestCase`. Không đổi framework các lớp cũ chỉ để pytest thu thập.

### 4.1. Ba yêu cầu chính của T1

Mỗi bộ chi phí dưới đây chạy đủ cả (a), (b), (c) bằng fixture pytest
tham số. S có phân phối đều trên các score được liệt kê; mỗi score k/d được
biểu diễn bằng d hàng với k nhãn dương, không có nhiễu Bernoulli.

| ID fixture | A,D,c | d; tử số k của hỗ trợ S | Vì sao |
|---|---|---|---|
| dyadic-ties | 8,2,.5 | 16; [0,1,2,11,12,13,16] | Bốn cặp cut tương đương ở hai điểm hòa |
| project-100-1-c0.2 | 100,1,.2 | 1000; [0,1,2,3,790,800,810,1000] | Kiểm trực tiếp chi phí dự án qua B1/B2/oracle, không chỉ công thức đóng |
| near-critical-c0.99 | 100,1,.99 | 20000; [0,196,198,199,200,201,20000] | c_crit=100/101; vùng E hẹp (.0099,.01), có score .00995 nằm trong vùng |
| empty-esc-c1 | 100,1,1 | 101; [0,1,2,101] | c>=c_crit; B1 phải trả hai ngưỡng trùng, B2 và oracle cùng risk nhị phân |

| Yêu cầu | Test tham số phụ trách | Đáp án độc lập |
|---|---|---|
| (a) B1 với a+=a-=c | `TestChowBaselines.test_t1a_b1_pietraszek_constant_abstention` | Risk chọn bằng bao dưới ba risk; công thức đóng được so với công thức suy riêng |
| (b) B2 cùng risk | `TestChowBaselines.test_t1b_b2_grid_same_risk` | `independent_grid_oracle` liệt kê tất cả cặp cut và tính risk trực tiếp, không gọi production helper |
| (c) Oracle regret=0 | `TestChowBaselines.test_t1c_independent_oracle_regret_zero` | R_S*=mean(min(A*s,c,D*(1-s))); R_2T* bằng oracle liệt kê riêng; thêm risk B1/B2 trừ R_S* bằng 0 |

**Hòa ở B1:** Với dyadic-ties, assert lo thuộc {1/16,1/8}, hi thuộc
{3/4,13/16}, và risk từng điểm bằng risk Bayes. Không khóa một trong bốn
cặp tương đương vào thứ tự duyệt của B1. Các fixture khác kiểm cut hợp lệ,
lo<=hi và risk tối thiểu; không ép ngưỡng thực nghiệm đúng giá trị giao
điểm khi có nhiều cut tương đương. Trường hợp E quá đắt kiểm lo=hi.

**Vì sao oracle độc lập:** Nếu expected lấy từ solver đang kiểm thì cùng một
lỗi có thể xuất hiện ở cả expected và actual. Các risk được viết trực tiếp
và phép liệt kê không gọi helper chọn action/cost trong source sản xuất.

### 4.2. Các ca bổ sung

| Test trong `TestChowConstant` | Mục đích/lý do |
|---|---|
| `test_asymmetric_cost_matrix` | A=7,D=2,c=.3; đủ 4 tổ hợp y,b để bắt đảo nhãn/thiếu phí |
| `test_textbook_asymmetric_chow` | Khóa đáp án .002,.8 theo chi phí mặc định |
| `test_class_dependent_constant_costs` | Kiểm mở rộng affine và risk tối thiểu từng điểm |
| `test_threshold_ties_and_neighbors` | `nextafter` ở hai phía kiểm dấu so sánh chính xác |
| `test_triple_tie_collapses_to_binary` | Kiểm dấu bằng tại chi phí tới hạn |
| `test_expensive_escalation_has_empty_region` | Không tạo E giả khi E quá đắt |
| `test_zero_cost_escalation` | Bao phủ c=0 và endpoint |
| `test_all_empty_bin_regions_roundtrip` | Mọi i<=j trên K=3, thêm biên K=1; bắt quy đổi cut 0/K sai |
| `test_exact_solver_all_pass_and_all_block` | Thực tế kiểm toàn PASS/E/B, bắt IndexError kL=n |
| `test_duplicate_scores_permutation_and_input_integrity` | Không tách score trùng, không phụ thuộc thứ tự, không sửa đầu vào |
| `test_invalid_closed_form_parameters` | Chặn các ca âm, NaN, chi phí lỗi bằng 0 đã chọn |

`TestBoundA1AndB` là test hỗ trợ cận trong `bound_B.md`, **không phải T3
của Thuỳ**, không thay thế `test_fragmented_regret.py`.

Dùng atol=rtol=1e-12 cho phép so số thực giải tích float64, chi phí cỡ vừa:
|actual-expected|<=atol+rtol*|expected|. atol xử lý gần 0; rtol xử lý scale.
Action và mảng so chính xác. Một số assertion dùng `assertAlmostEqual`
hoặc equality có chủ đích. Không dùng `isclose` mở rộng điểm hòa thành dải.
`nextafter` kiểm phép so sánh, không đòi chênh risk một ULP luôn đo được.
Không áp tolerance này cho sai số lấy mẫu, hiệu chuẩn, chia bin hoặc tích
phân; phải phân tích riêng nếu đổi scale hoặc gần suy biến.

## Checklist rà soát T3 của Thuỳ

File đúng theo phân công: `tests/test_fragmented_regret.py`. Chưa có trong
checkout được rà; tất cả mục dưới là **chờ đối chiếu**, không ghi pass.

- [ ] Fixture có ít nhất hai khúc ESC tách nhau bằng vùng không ESC; ghi
      action sequence và biên các khúc, không chỉ assert số action.
- [ ] R_S* và R_2T* được suy/tính độc lập từ fixture; không lấy output thuật
      toán đang kiểm làm expected. Có đáp án cụ thể và đơn vị risk.
- [ ] Delta=R_2T*-R_S*>0 và khớp đáp án theo tolerance có giải thích.
- [ ] Delta<=B+epsilon với B đúng giả thiết; đồng thời kiểm cận A1:
      Delta<=R_A1-R_S*+epsilon. Có thể kiểm min hai cận.
- [ ] Ghi rõ phân phối S, trọng số chuẩn hóa tổng 1, rời rạc hay liên tục;
      nếu tích phân số, tách sai số tích phân khỏi tolerance float.
- [ ] Ngưỡng oracle cho phép vùng E rỗng; score trùng không bị tách;
      tie được xử lý nhất quán hoặc đối chiếu qua risk tối ưu.

**Vì sao:** Một phản ví dụ có Delta dương không tự kiểm được cận hay tính
độc lập oracle. T3 cần đo đúng regret của hạn chế hai ngưỡng trên cùng phân
phối, không trộn thêm chênh lệch giữa hai estimator học từ mẫu.

## Checklist dữ liệu bắt buộc theo đề

- [ ] Scaler, mô hình và calibrator chỉ fit trên train; Platt dùng CV nội
      bộ train (score ngoài fold). Val chỉ chọn ngưỡng; test chỉ đánh giá.
- [ ] Không bỏ dòng trùng để thay cho gom nhóm. Gom các hàng có vector
      đặc trưng bằng nhau theo giá trị float64 vào cùng tập, không so chuỗi
      CSV và không dùng nhãn trong khóa nhóm. Credit Card Fraud không có ID.
- [ ] “Chuẩn hóa” ở đây chỉ là chuẩn hóa **cách biểu diễn số**: parse các cột
      sang float64, quy ước biểu diễn nhất quán; không phải scaling đặc trưng,
      không fit tham số gì và không phát sinh rò rỉ. Ví dụ chuỗi 1 và 1.0 cùng
      giá trị. Không tự làm tròn để gom cả các giá trị gần nhau; xác thực
      NaN/Inf trước khi lập khóa. Scaler dùng cho mô hình là bước riêng chỉ
      fit train, không liên quan khóa nhóm trùng.
- [ ] Manifest ghi số hàng, số nhóm trùng, số hàng thuộc nhóm trùng và số
      bản dư sum(size-1); phân bố nhóm theo split; kiểm không nhóm xuyên tập.
- [ ] Miền nhãn/b, độ dài, NaN/Inf, score [0,1] hợp lệ; solver không sửa mảng.

**Lỗi cần Phước xử lý:** Commit gốc `scripts/run_creditcard.py` fit Platt trên
val (dòng 76) và `drop_duplicates` (dòng 59); `tests/test_roc.py:27` cũng fit
calibrator trên val. Đây là sai đặc tả, không phải lựa chọn thiết kế được
chấp nhận. Chưa sửa pipeline trong phần việc T1; xem `group_handoff.md`.
Không dùng log unit test để xác nhận tuân thủ dữ liệu thực.

## Lệnh nghiệm thu

Cài phụ thuộc test bằng `python -m pip install -r requirements-dev.txt`.
Chạy từ thư mục gốc repository:

```bash
python -m pytest tests/ -q
python -m pytest tests/test_chow_constant.py -q
```

Log phải chứa SHA commit chạy, lệnh, phiên bản môi trường, exit code và
stdout/stderr thật. Chỉ ghi pass sau khi có log, không dùng kết quả unittest
cũ thay log pytest. Chứng minh và giới hạn cận: `bound_B.md`.
