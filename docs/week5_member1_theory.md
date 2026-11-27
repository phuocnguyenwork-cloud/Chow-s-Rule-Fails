> Lưu trữ audit/thực nghiệm ngày 20/11 trên nền 8e87cc9. Không phải đánh giá main hiện tại; xem `week5_merge_20261127.md` cho trạng thái sau tích hợp fix của Phước.

# Tuần 5 — Thành viên 1: khóa và kiểm chứng phần lý thuyết

**Thời gian:** 16/11–22/11/2026  
**Phạm vi:** chỉ rà soát lý thuyết đã có; không bổ sung tính năng, mô hình hay tập dữ liệu.  
**Git HEAD khi kiểm tra:** `8e87cc9`.

## 1. Kết luận rà soát

Các định nghĩa trong `theory.md`, `theory_2.md` và `docs/midterm_theory_section.md` thống nhất với mã nguồn:

- score tăng theo xác suất lớp nguy hiểm/gian lận;
- thứ tự hành động là PASS (0), ESCALATE (1), BLOCK (2);
- chi phí điều kiện là
  \[
  r_P(s)=As,\qquad r_B(s)=D(1-s),\qquad
  r_E(s)=c+As\,m(s)+D(1-s)f(s);
  \]
- ESCALATE gồm cả phí gọi tầng 2 và chi phí lỗi quyết định của tầng 2;
- lớp hai ngưỡng cho phép vùng ESCALATE rỗng (`t_L=t_H`), vì vậy chứa lớp PASS/BLOCK một ngưỡng.

Với Chow cổ điển, hai ngưỡng đã được dẫn xuất đúng:

\[
t_L=c/A,\qquad t_H=1-c/D
\]

khi \(c<AD/(A+D)\). Nếu không, nghiệm suy biến là
\(t_L=t_H=D/(A+D)\). Với cấu hình dự án \((A,D,c)=(100,1,0.2)\), kết quả là \((0.002,0.8)\).

Với chi phí ESCALATE affine theo lớp
\(e_0(s)=a_+s+a_-(1-s)\), tài liệu đã có điều kiện chính xác để vùng ESCALATE có độ dài dương:

\[
\frac{a_+}{A}+\frac{a_-}{D}<1.
\]

Nếu điều kiện không đúng, vẫn tồn tại nghiệm Bayes thuộc lớp hai ngưỡng nhưng vùng ESCALATE rỗng.

## 2. Điều kiện interval-optimality được phép dùng trong tuần 5

Đối với risk ESCALATE tổng quát \(e(s)\), một chính sách hai ngưỡng là Bayes tối ưu **khi và chỉ khi** tồn tại \(t_L\le t_H\) sao cho, gần khắp nơi theo phân phối của \(S\):

\[
As\le \min\{e(s),D(1-s)\}\quad (s<t_L),
\]
\[
e(s)\le \min\{As,D(1-s)\}\quad (t_L\le s<t_H),
\]
\[
D(1-s)\le \min\{As,e(s)\}\quad (s\ge t_H).
\]

Đây là đặc tả chính xác theo bao dưới của ba đường risk và xử lý được cả điểm hòa. Dạng kiểm tra tương đương trên lưới bin là: tồn tại một nghiệm `argmin` theo từng bin có chuỗi hành động không giảm `PASS* ESCALATE* BLOCK*`.

Một điều kiện cấu trúc thuận tiện để khảo sát vùng ESCALATE là đặt

\[
g_P(s)=e(s)-As,\qquad g_B(s)=e(s)-D(1-s).
\]

Vùng mà ESCALATE là tối ưu là giao của \(\{g_P\le0\}\) và \(\{g_B\le0\}\). Nếu \(g_P\) không tăng và \(g_B\) không giảm, giao này là một khoảng (có thể rỗng). Tuy nhiên, để kết luận toàn bộ chính sách có đúng thứ tự PASS–ESCALATE–BLOCK vẫn phải kiểm các bất đẳng thức bao dưới ở trên và quy ước điểm hòa.

**Giới hạn đã khóa:** tính lồi/lõm của ROC tầng 1, đứng riêng, không xác định \(m(s)\) và \(f(s)\), nên không đủ để suy ra interval-optimality khi tầng 2 phụ thuộc score. Tuần 5 không tuyên bố một định lý ROC mạnh hơn khi chưa có chứng minh. Đây là điểm cần ghi nhất quán trong báo cáo và thuyết trình.

## 3. Regret đã được chứng minh

Đặt \(R_S^*\) là risk tối ưu trong mọi chính sách dùng score, \(R_{2T}^*\) là risk tối ưu trong lớp hai ngưỡng, và
\(\Delta=R_{2T}^*-R_S^*\). Vì lớp hai ngưỡng chứa chính sách PASS/BLOCK một ngưỡng tốt nhất có risk \(R_{A1}\):

\[
0\le\Delta\le R_{A1}-R_S^*.
\]

Với surrogate affine \(e_0\), nếu chỉ cột risk ESCALATE thay đổi, \(\delta(s)=e(s)-e_0(s)\), và một nghiệm Bayes của surrogate thuộc lớp hai ngưỡng, thì:

\[
0\le\Delta\le \mathbb E|\delta(S)|.
\]

Khi cả hai cận áp dụng:

\[
0\le\Delta\le
\min\left\{R_{A1}-R_S^*,\ \mathbb E|e(S)-e_0(S)|\right\}.
\]

Các cận này là phát biểu ở mức phân phối. Chênh lệch test của hai chính sách học từ validation có thể âm do nhiễu ước lượng và không mâu thuẫn với \(\Delta\ge0\).

## 4. Phản ví dụ đã chốt

Fixture 5 bin trong `tests/test_fragmented_regret.py` cho chuỗi tối ưu
`PASS, ESCALATE, BLOCK, ESCALATE, BLOCK`, không thuộc lớp hai ngưỡng. Các giá trị kiểm tay:

- \(R_S^*=0.18576\);
- \(R_{2T}^*=0.19149\);
- \(\Delta=0.00573>0\);
- \(R_{A1}=0.26450\), nên cận A1 bằng \(0.07874\).

Đối chứng miss-rate phẳng cho chuỗi `PASS, ESCALATE, ESCALATE, ESCALATE, BLOCK` và regret bằng 0 trên cùng fixture.

## 5. Bằng chứng kiểm thử

Lệnh đã chạy:

```bash
python -m pytest -q tests/test_chow_constant.py tests/test_fragmented_regret.py
```

Kết quả: **53 passed** trong 0,49 giây. Có một cảnh báo pytest không ghi được cache do filesystem chỉ đọc; cảnh báo không ảnh hưởng kết quả kiểm thử.

Các kiểm thử bao phủ: ma trận chi phí bất đối xứng, Chow cổ điển, chi phí affine theo lớp, vùng ESCALATE rỗng, điểm hòa, exact solver, hai cận regret, phản ví dụ nhiều khoảng và 20 bảng cost ngẫu nhiên.

## 6. Việc không làm do Scope Lock

- Không mở rộng định lý sang ROC-convexity khi chưa đủ giả thiết về lỗi tầng 2 theo score.
- Không thay đổi mã solver, pipeline, mô hình hay tham số thực nghiệm.
- Không biến cận ở mức phân phối thành khoảng tin cậy hữu hạn mẫu nếu chưa có phân tích thống kê tương ứng.
- Không gọi regret test là “Bayes regret thật”; trong thực nghiệm đó là chênh lệch giữa các chính sách học/đánh giá theo thiết kế bin đã khóa.

## 7. Rủi ro còn lại cần ghi trong báo cáo chung

1. Tiêu đề đề tài nói “exact conditions”; bản thảo hiện có điều kiện chính xác theo bao dưới risk và kết quả đóng cho trường hợp affine, nhưng chưa có đặc trưng đơn giản bằng ROC cho tầng 2 tổng quát.
2. Các ước lượng \(m(s),f(s)\) theo bin có thể nhiễu ở bin ít gian lận; không nên diễn giải mọi đứt đoạn quan sát được là cấu trúc tổng thể.
3. Cần giữ cùng score, cùng phân phối và cùng định nghĩa chi phí khi báo regret; nếu oracle dùng thêm đặc trưng, chênh lệch không còn chỉ đo hạn chế hai ngưỡng.
