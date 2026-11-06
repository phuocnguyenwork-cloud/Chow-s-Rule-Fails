# LÝ THUYẾT CHÍNH SÁCH RỦI RO BA HÀNH ĐỘNG

## Nghiệm Chow, quy ước score và chi phí, tính tối ưu hai ngưỡng và cận regret

**Đề tài:** G5-1 — *Chow’s Rule Fails When the Abstention Is a Classifier: Exact Conditions for Interval-Optimality of Three-Action Risk Policies, and the Regret of Two-Threshold Designs.*  
**Phạm vi bàn giao:** Phần lý thuyết và đặc tả kiểm chứng của An.  
**Ngày tổng hợp:** 05/11/2026.  
**Phiên bản mã dùng để đối chiếu:** `14bd0342a7d348ee0f23cdbcf806c82815bfbca6`.

> Báo cáo thống nhất các quy ước và suy luận đã chốt: score tăng theo rủi ro lớp 1; chi phí sai phân lớp được phép bất đối xứng; hành động ESCALATE chịu cả phí gọi và lỗi tầng 2; lớp hai ngưỡng cho phép vùng chuyển tầng rỗng. Nghiệm Chow được dẫn xuất từ ba đường rủi ro. Cận regret kết hợp là $0\le\Delta\le\min\{B,R_{A1}-R_S^*\}$ khi các giả thiết của cận B được thỏa mãn. Điều kiện cần và đủ tổng quát cho tầng 2 phụ thuộc score và việc đối chiếu bản chứng minh gốc của nhóm vẫn là phần còn mở.

## Mục lục

1. Bối cảnh và câu hỏi nghiên cứu.
2. Ký hiệu, score và phạm vi thông tin.
3. Ma trận chi phí và rủi ro điều kiện.
4. Chính sách Bayes và các lớp chính sách.
5. Nghiệm Chow với chi phí bất đối xứng.
6. Chi phí chuyển tầng hằng theo lớp.
7. Tầng 2 phụ thuộc score và khả năng nhiều khoảng.
8. Quy ước hòa, score trùng và vùng rỗng.
9. Cận sơ bộ từ chính sách A1.
10. Cận B: giả thiết, phát biểu và chứng minh.
11. Cận theo sai lệch lỗi tầng 2 và giới hạn áp dụng.
12. Cơ sở lý thuyết của kiểm thử Chow T1.
13. Tiêu chí rà soát T3 của Thuỳ.
14. Sai số số học và tính nguyên vẹn dữ liệu.
15. Quan hệ với mã nguồn, kiểm chứng và phần còn mở.
16. Kết luận và nguồn tổng hợp.

## 1. Bối cảnh và câu hỏi nghiên cứu

Một hệ thống phân tầng có thể xử lý một đầu vào theo ba cách: ra quyết định cho qua ngay, chuyển sang một bộ phân lớp cấp hai, hoặc chặn ngay. Mục tiêu là giảm chi phí kỳ vọng của cả hệ thống, bao gồm lỗi quyết định cuối và tài nguyên dành cho chuyển tầng.

Một thiết kế dễ triển khai chia score thành ba vùng liên tiếp: PASS → ESCALATE → BLOCK. Tuy nhiên, tính đơn giản của thiết kế không tự bảo đảm tính tối ưu. Câu hỏi là: khi nào tồn tại chính sách Bayes có dạng hai ngưỡng, và nếu giới hạn vào hai ngưỡng làm mất tính tối ưu thì phần chi phí tăng thêm được chặn như thế nào?

Trường hợp Chow chuẩn sử dụng chi phí abstention hằng. Khi chuyển tầng là gọi một bộ phân lớp bất toàn, chi phí này có thể thay đổi theo vùng score do lỗi của bộ phân lớp thay đổi. Do đó, cần phân biệt ba mô hình:

| Mô hình | Rủi ro chuyển tầng | Hệ quả lý thuyết trong phạm vi báo cáo |
|---|---|---|
| Chow chuẩn | Hằng $c$ | Có nghiệm hai ngưỡng hoặc nghiệm suy biến một ngưỡng |
| Hằng theo lớp | $a_+s+a_-(1-s)$ | Vẫn có nghiệm hai ngưỡng, bao gồm vùng ESC rỗng |
| Tầng 2 phụ thuộc score | Hàm $e(s)$ tổng quát | Có thể tạo nhiều khoảng; cần kiểm tra cấu trúc risk |

Tên đề tài diễn đạt giới hạn của việc mở rộng giả thiết Chow sang tầng 2 bất toàn. Nó không có nghĩa định lý Chow sai dưới chính các giả thiết ban đầu.

## 2. Ký hiệu, score và phạm vi thông tin

### 2.1. Nhãn và hành động

Quy ước nhãn và mã hành động:

| Ký hiệu | Giá trị hoặc ý nghĩa |
|---|---|
| $Y=1$ | Gian lận hoặc nguy hiểm |
| $Y=0$ | Hợp lệ |
| $S$ | Score đầu ra tầng 1, giá trị hữu hạn trong $[0,1]$ |
| $P$ / PASS / mã 0 | Cho qua, quyết định lớp 0 |
| $E$ / ESCALATE / mã 1 | Gọi tầng 2 ra quyết định cuối |
| $B$ / BLOCK / mã 2 | Chặn, quyết định lớp 1 |

Từ “Reject” trong chính sách của dự án được hiểu là BLOCK. Cần ghi rõ vì trong một số cách dùng của reject option, “reject” lại chỉ hành động abstain.

### 2.2. Quy ước score đã chốt

**Score càng lớn thì nguy cơ thuộc lớp 1 càng cao.** Khi dẫn xuất công thức ngưỡng xác suất, giả sử score hiệu chuẩn trong tổng thể:

$$
\mathbb P(Y=1\mid S=s)=s.
$$

Nhờ giả thiết này, xác suất điều kiện của lớp 0 là $1-s$. Mỗi hàng ma trận chi phí được đưa vào kỳ vọng với đúng trọng số xác suất tương ứng.

Nếu score chưa hiệu chuẩn, đặt:

$$
\eta(s)=\mathbb P(Y=1\mid S=s).
$$

Khi đó phải dùng $\eta(s)$ trong các biểu thức risk. Không thể mặc định một score xếp hạng tốt có ý nghĩa xác suất chính xác. Nếu $\eta$ không đơn điệu theo score thô, một ngưỡng trên posterior cũng chưa chắc tương ứng với một ngưỡng trên score thô.

**Lý do chọn quy ước:** Nó khớp với nghĩa nhãn và chi phí trong code; tránh đảo chiều PASS/BLOCK; đồng thời giúp giải thích các ngưỡng bằng xác suất gian lận. Việc dùng Platt trong pipeline là một phương pháp ước lượng hiệu chuẩn, không phải chứng minh hiệu chuẩn hoàn hảo trên tổng thể.

### 2.3. Mọi chính sách so sánh phải dùng cùng thông tin

Báo cáo xét chính sách chỉ là hàm của score $S$. Tầng 2 có thể dùng thông tin phong phú hơn, nhưng người chọn hành động P/E/B ở tầng định tuyến chỉ quan sát score đã quy định.

Nếu oracle được phép quan sát toàn bộ đặc trưng $X$ còn chính sách hai ngưỡng chỉ quan sát $S$, chênh lệch chi phí chứa cả tổn thất do mất thông tin. Nó không còn chỉ phản ánh hạn chế hình dạng hai ngưỡng. Vì vậy phải giữ cùng phạm vi thông tin trong mọi phát biểu regret ở dưới.

## 3. Ma trận chi phí và rủi ro điều kiện

### 3.1. Chi phí bất đối xứng

Đặt:

$$
A=C_{FN}>0,\qquad D=C_{FP}>0,\qquad c\ge0.
$$

$A$ là chi phí cho qua một gian lận; $D$ là chi phí chặn một giao dịch hợp lệ; $c$ là phí gọi tầng 2. Tất cả dùng cùng đơn vị chi phí trên một giao dịch. Quyết định đúng có chi phí bằng 0.

Với tầng 2 hoàn hảo, ma trận chi phí là:

| Nhãn thật | PASS | ESCALATE | BLOCK |
|---|---:|---:|---:|
| $Y=0$ | $0$ | $c$ | $D$ |
| $Y=1$ | $A$ | $c$ | $0$ |

Dùng $A\ne D$ vừa phản ánh sự khác nhau giữa hai loại lỗi, vừa giúp kiểm thử phát hiện lỗi đảo nhãn hoặc áp dụng công thức đối xứng không phù hợp. Bộ $100/1/0.2$ là giả định thí nghiệm của dự án, không phải tỷ lệ suy ra từ dữ liệu.

Đây là lớp ma trận có chi phí quyết định đúng bằng 0. Báo cáo không tự mở rộng các công thức cho mọi ma trận chi phí 2×3 tùy ý.

### 3.2. Tầng 2 bất toàn

Gọi $b\in\{0,1\}$ là quyết định cuối của tầng 2, trong đó $b=1$ nghĩa là BLOCK. Chi phí trên một giao dịch là:

$$
\ell_P(y)=Ay,\qquad \ell_B(y)=D(1-y),
$$

$$
\ell_E(y,b)=c+Ay(1-b)+D(1-y)b.
$$

Phí gọi $c$ phải trả dù tầng 2 quyết định đúng hay sai. Nếu chỉ tính $c$, hệ thống sẽ đánh giá thiếu chi phí chuyển tầng thực tế.

Đặt hai tỷ lệ lỗi điều kiện:

$$
m(s)=\mathbb P(b=0\mid Y=1,S=s),
$$

$$
f(s)=\mathbb P(b=1\mid Y=0,S=s).
$$

Với score hiệu chuẩn, ba risk điều kiện là:

$$
\boxed{r_P(s)=As,\qquad r_B(s)=D(1-s),}
$$

$$
\boxed{r_E(s)=c+As\,m(s)+D(1-s)f(s).}
$$

Ví dụ ma trận kiểm tra trực tiếp với $A=7,D=2,c=0.3$:

| $y$ | $b$ | PASS | ESCALATE | BLOCK |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0.3 | 2 |
| 0 | 1 | 0 | 2.3 | 2 |
| 1 | 0 | 7 | 7.3 | 0 |
| 1 | 1 | 7 | 0.3 | 0 |

Đủ bốn tổ hợp giúp phát hiện thiếu phí gọi, thiếu một loại lỗi hoặc đảo nghĩa `tier2_block`.

## 4. Chính sách Bayes và các lớp chính sách

### 4.1. Risk tổng thể và nghiệm Bayes theo score

Với chính sách $\pi$, định nghĩa:

$$
R(\pi)=\mathbb E\big[r_{\pi(S)}(S)\big].
$$

Gọi $\Pi_S$ là lớp chính sách đo được chỉ dùng $S$. Không có ràng buộc ngân sách chuyển tầng, nghiệm Bayes có thể chọn độc lập theo từng score:

$$
\pi_S^*(s)\in\operatorname*{arg\,min}_{a\in\{P,E,B\}}r_a(s),
$$

$$
R_S^*=\mathbb E\left[\min\{r_P(S),r_E(S),r_B(S)\}\right].
$$

### 4.2. Lớp hai ngưỡng và lớp A1

Lớp $\Pi_{2T}$ gồm các chính sách:

$$
\pi_{t_L,t_H}(s)=
\begin{cases}
P,&s<t_L,\\
E,&t_L\le s<t_H,\\
B,&s\ge t_H,
\end{cases}
\qquad t_L\le t_H.
$$

Cho phép ngưỡng mở rộng bằng $\pm\infty$ để biểu diễn các nghiệm chỉ dùng một hành động.

Lớp $\Pi_{A1}$ gồm chính sách PASS/BLOCK một ngưỡng. Mỗi chính sách A1 là một chính sách hai ngưỡng với $t_L=t_H$. Do đó:

$$
\Pi_{A1}\subseteq\Pi_{2T}\subseteq\Pi_S.
$$

Đặt:

$$
R_{2T}^*=\inf_{\pi\in\Pi_{2T}}R(\pi),\qquad
\Delta=R_{2T}^*-R_S^*.
$$

Dùng infimum để không phải thêm giả thiết tồn tại nghiệm tối thiểu trong mọi mô hình tổng quát. Nếu tối thiểu đạt được, có thể viết $R_{2T}^*=R(\pi_{2T}^*)$. Trên các fixture hữu hạn, tập cặp cut là hữu hạn nên nghiệm tối thiểu tồn tại.

**Ý nghĩa:** $\Delta$ đo giá của việc giới hạn chính sách vào hai ngưỡng, với cùng thông tin và cùng phân phối; không phải chênh lệch tùy ý giữa hai mô hình học được.

## 5. Nghiệm Chow với chi phí bất đối xứng

### 5.1. Dẫn xuất hai ngưỡng

Trong trường hợp Chow chuẩn, $r_E(s)=c$. Giải giao điểm PASS/ESCALATE:

$$
As=c\quad\Longrightarrow\quad t_L=\frac cA.
$$

Giải giao điểm ESCALATE/BLOCK:

$$
c=D(1-s)\quad\Longrightarrow\quad t_H=1-\frac cD.
$$

Nếu các ngưỡng theo đúng thứ tự, vùng score trung gian là nơi E không đắt hơn cả P và B.

### 5.2. Điều kiện vùng ESC có độ dài dương

$$
t_L<t_H
\iff \frac cA<1-\frac cD
\iff c<\frac{AD}{A+D}.
$$

Đặt:

$$
c_{\mathrm{crit}}=\frac{AD}{A+D},\qquad t_0=\frac D{A+D}.
$$

Kết quả:

$$
\boxed{
(t_L,t_H)=
\begin{cases}
(c/A,\;1-c/D),&0\le c<c_{\mathrm{crit}},\\
(t_0,t_0),&c\ge c_{\mathrm{crit}}.
\end{cases}}
$$

Ở trường hợp thứ hai, chọn nghiệm nhị phân PASS/BLOCK vì E không tạo khoảng có risk thấp hơn cả hai hành động trực tiếp. Tại $c=c_{\mathrm{crit}}$, cả ba hòa ở riêng $t_0$; chọn BLOCK tại đó vẫn Bayes tối ưu.

Không sửa hai giao điểm đảo thứ tự bằng cách hoán đổi hoặc chỉ clip về $[0,1]$: các thao tác đó không xuất phát từ việc chọn risk nhỏ nhất.

### 5.3. Ví dụ của dự án

Với $A=100,D=1,c=0.2$:

$$
t_L=0.002,\qquad t_H=0.8.
$$

Ngưỡng không đối xứng quanh 0.5. Bỏ sót gian lận có mức phạt lớn nên một xác suất gian lận nhỏ cũng có thể khiến PASS đắt hơn gọi tầng 2.

Với $c=0.99$:

$$
c<c_{\mathrm{crit}}=\frac{100}{101},\qquad
(t_L,t_H)=(0.0099,0.01).
$$

Vùng E rất hẹp nhưng còn tồn tại. Với $c=1$, chọn hai ngưỡng trùng bằng $1/101$.

“Vùng có độ dài dương” là thuộc tính của khoảng trên trục score. Phân phối cụ thể vẫn có thể không đặt khối lượng xác suất vào khoảng đó; vì vậy không đồng nhất độ rộng khoảng với tỷ lệ giao dịch được chuyển tầng.

## 6. Chi phí chuyển tầng hằng theo lớp

### 6.1. Risk affine

Giả sử chi phí E có điều kiện theo lớp là các hằng không âm $a_+,a_-$. Khi đó:

$$
e_0(s)=a_+s+a_-(1-s).
$$

Nếu các tỷ lệ lỗi tầng 2 là hằng $m_0,f_0$, có thể đặt:

$$
a_+=c+Am_0,\qquad a_-=c+Df_0.
$$

Chow chuẩn là trường hợp đặc biệt $a_+=a_-=c$.

### 6.2. Công thức và điều kiện

Giải $As=e_0(s)$ và $e_0(s)=D(1-s)$ cho:

$$
t_L=\frac{a_-}{A-a_++a_-},\qquad
t_H=\frac{D-a_-}{D-a_-+a_+}.
$$

Vùng E có độ dài dương khi và chỉ khi:

$$
\boxed{\frac{a_+}{A}+\frac{a_-}{D}<1.}
$$

Trong miền này, các mẫu số hợp lệ và các giao điểm cho bao dưới theo thứ tự P,E,B. Nếu tổng lớn hơn hoặc bằng 1, chọn $(t_0,t_0)$ thay cho hai công thức giao điểm.

### 6.3. Vì sao điều kiện này đúng?

Tại giao điểm P/B là $t_0=D/(A+D)$:

$$
e_0(t_0)=\frac{Da_++Aa_-}{A+D},\qquad
r_P(t_0)=r_B(t_0)=\frac{AD}{A+D}.
$$

So sánh hai biểu thức thu được điều kiện trên. Nếu E thấp hơn ở $t_0$, nhờ tính affine, nó tạo phần trung gian của bao dưới. Nếu E cao hơn hoặc bằng ở $t_0$, trên đoạn $[0,t_0]$ hiệu $e_0-r_P$ có hai đầu không âm; trên đoạn $[t_0,1]$ hiệu $e_0-r_B$ cũng có hai đầu không âm. Tính affine suy ra E không thấp hơn risk nhị phân ở mọi score.

**Hệ quả quan trọng:** Với mọi $a_+,a_-\ge0$, luôn chọn được một nghiệm Bayes surrogate thuộc $\Pi_{2T}$, bao gồm nghiệm suy biến $(t_0,t_0)$. Không cần giả thiết vùng ESC có độ dài dương khi dùng surrogate này trong chứng minh cận B.

## 7. Tầng 2 phụ thuộc score và khả năng nhiều khoảng

Khi $m(s),f(s)$ thay đổi theo score, risk E là:

$$
e(s)=c+As\,m(s)+D(1-s)f(s).
$$

Một vùng tầng 2 có miss rate cao có thể làm E đắt hơn BLOCK, trong khi ở các vùng lân cận E lại rẻ hơn. Hình dạng nghiệm có thể gồm E → B → E, dẫn tới các khúc ESC tách nhau.

Cách suy luận đúng là xét hai hiệu:

$$
g_P(s)=e(s)-As,\qquad g_B(s)=e(s)-D(1-s).
$$

E là một hành động Bayes tại các score mà đồng thời $g_P(s)\le0$ và $g_B(s)\le0$. Khi xét tính tối ưu nghiêm ngặt, thay hai dấu bằng dấu nhỏ hơn. Cấu trúc đổi dấu của các hiệu quyết định vùng E có liền nhau hay không.

Lỗi trung bình theo lớp không đủ mô tả các hiệu này. Một mô hình hằng theo lớp có thể phù hợp về trung bình nhưng che khuất vùng tầng 2 hoạt động kém.

B1 dựa trên ROC và giả thiết chi phí hằng theo lớp là baseline hợp lệ. Tuy nhiên, ROC tầng 1 không tự chứa thông tin về các lỗi điều kiện của tầng 2. Không thể chỉ từ tính lõm của biên trên ROC hull mà kết luận mọi hệ thống tầng 2 đều có nghiệm hai ngưỡng tối ưu. “Convex hull” cũng không có nghĩa hàm TPR theo FPR trên biên trên phải lồi; biên trên thường được mô tả bằng độ dốc không tăng.

Báo cáo dừng ở cơ sở phân tích này; chưa tuyên bố một định lý điều kiện cần và đủ cho hàm $e(s)$ tổng quát.

## 8. Quy ước hòa, score trùng và vùng rỗng

### 8.1. Gán biên trong chính sách ngưỡng

| Trường hợp | Quyết định |
|---|---|
| $s<t_L$ | PASS |
| $t_L\le s<t_H$ | ESCALATE |
| $s\ge t_H$ | BLOCK |
| $t_L=t_H=t$ | PASS khi $s<t$, BLOCK khi $s\ge t$ |
| $(t_L,t_H)=(-\infty,-\infty)$ | Toàn BLOCK |
| $(t_L,t_H)=(+\infty,+\infty)$ | Toàn PASS |
| $(t_L,t_H)=(-\infty,+\infty)$ | Toàn ESCALATE với score hữu hạn |

Gán điểm bằng ngưỡng sang vùng bên phải để tương thích hàm ngưỡng và cách gán bin đang dùng. Đây là quy ước tái lập; tối ưu Bayes không buộc chọn hành động mạnh hơn tại điểm hòa.

Nếu hai hành động có cùng risk thật ở một score, đổi giữa chúng không làm thay đổi risk kỳ vọng, kể cả khi phân phối có khối lượng tại chính score đó. Nếu các risk chỉ gần nhau vì sai số ước lượng, không được tự coi đó là hòa thật.

### 8.2. Các bộ giải có thể phá hòa khác nhau

`per_bin_actions` dùng `np.argmin`, nên ưu tiên chỉ số nhỏ hơn, tức P trước E trước B. Luật ngưỡng gán biên sang phải. Hai lựa chọn này có thể khác action tại điểm hòa nhưng cùng risk tối ưu.

Vì vậy khi kiểm nghiệm tối ưu cần phân biệt:

- Kiểm luật dự đoán cụ thể: action phải đúng dấu so sánh đã quy định.
- So hai nghiệm Bayes đồng hạng: kiểm risk và tập nghiệm hợp lệ, không ép cùng một cặp cut nếu đặc tả cho phép nhiều cặp.

Bộ quét theo bin đang dùng điều kiện cải thiện tổng cost lớn hơn `1e-12`; exact solver dùng so sánh strict. Không mặc định chúng có cùng thứ tự ưu tiên mọi nghiệm gần đồng hạng.

### 8.3. Score trùng và bin

Một chính sách chỉ là hàm của S phải cho cùng action trên các hàng có cùng score. Tách các hàng đó giữa hai action sẽ sử dụng thêm thông tin ngoài score và thay đổi lớp chính sách đang đánh giá.

Với K bin, cho phép $0\le i\le j\le K$: PASS trước i, E từ i đến trước j và BLOCK từ j trở đi. Các trường hợp $i=j$, $i=0$, $j=K$ đều hợp lệ. Không ép E phải chứa ít nhất một bin.

## 9. Cận sơ bộ từ chính sách A1

Cho $R_{A1}$ là risk của một chính sách PASS/BLOCK một ngưỡng khả thi; có thể chọn chính sách tốt nhất trong lớp này. Ký hiệu $R_0$ của đề được hiểu theo nghĩa đó.

Từ quan hệ bao hàm lớp chính sách:

$$
R_S^*\le R_{2T}^*\le R_{A1}.
$$

Trừ $R_S^*$:

$$
\boxed{0\le\Delta\le R_{A1}-R_S^*.}
$$

**Vì sao phải cho vùng E rỗng:** Khi $t_L=t_H$, chính sách hai ngưỡng thực hiện được chính xác một chính sách A1. Nếu ép $t_L<t_H$, lập luận bao hàm này không còn trực tiếp đúng theo cùng cách.

Cận A1 không cần giả thiết surrogate của cận B ở mục sau. Tuy nhiên, các risk so sánh phải cùng phân phối và cùng định nghĩa chi phí; không thay chúng bằng các ước lượng trên các mẫu khác nhau rồi coi bất đẳng thức là tự động đúng hữu hạn mẫu.

## 10. Cận B: giả thiết, phát biểu và chứng minh

### 10.1. Giả thiết

So sánh risk thật $r$ với risk surrogate $r^0$, trong đó chỉ action E thay đổi:

$$
r_P=r_P^0,\qquad r_B=r_B^0,
$$

$$
r_E(s)=e(s),\qquad r_E^0(s)=e_0(s),\qquad
\delta(s)=e(s)-e_0(s).
$$

Giả sử:

1. Các chính sách dùng cùng score S và được đánh giá trên cùng phân phối.
2. Các risk khả tích; $\mathbb E|\delta(S)|$ hữu hạn để cận hữu ích.
3. Không có ràng buộc ngân sách chuyển tầng ghép các quyết định giữa các đầu vào.
4. Có một nghiệm Bayes surrogate $\pi_0$ thuộc lớp hai ngưỡng, với cách chọn hòa phù hợp lớp này.

Điều kiện 4 được bảo đảm cho surrogate hằng hoặc affine không âm theo mục 6, với risk P/B đã xác định bằng score hiệu chuẩn. Nó vẫn đúng khi nghiệm là $(t_0,t_0)$.

### 10.2. Phát biểu

Đặt:

$$
B=\mathbb E|e(S)-e_0(S)|.
$$

Khi đó:

$$
\boxed{
0\le\Delta
\le R(\pi_0)-R_S^*
\le B.}
$$

### 10.3. Chứng minh từng bước

Vì $\pi_0\in\Pi_{2T}$:

$$
R_S^*\le R_{2T}^*\le R(\pi_0).
$$

Tại mỗi score s, tính tối ưu Bayes surrogate cho:

$$
r^0_{\pi_0(s)}(s)-r^0_{\pi_S^*(s)}(s)\le0.
$$

Chỉ risk E thay đổi, do đó:

$$
\begin{aligned}
r_{\pi_0(s)}(s)-r_{\pi_S^*(s)}(s)
&=r^0_{\pi_0(s)}(s)-r^0_{\pi_S^*(s)}(s)\\
&\quad+\delta(s)\left(
\mathbf 1\{\pi_0(s)=E\}-\mathbf 1\{\pi_S^*(s)=E\}
\right)\\
&\le |\delta(s)|.
\end{aligned}
$$

Hiệu hai chỉ báo chỉ có thể bằng −1, 0 hoặc 1. Lấy kỳ vọng thu được:

$$
R(\pi_0)-R_S^*\le\mathbb E|\delta(S)|=B.
$$

Ghép các bất đẳng thức hoàn tất chứng minh. Không cần nghiệm surrogate duy nhất: tại hòa, chỉ cần chọn một nghiệm Bayes surrogate khả thi trong lớp hai ngưỡng.

### 10.4. Vì sao hệ số là 1?

Nếu tất cả action risk đều có thể sai tối đa $\varepsilon$, phép so hai action có thể cộng hai sai số và dẫn đến cận tổng quát $2\varepsilon$. Trong mô hình đang xét, chỉ một cột E thay đổi. Hiệu nhiễu có dạng $\delta$ nhân hiệu hai chỉ báo, nên không vượt quá $|\delta|$.

Do đó:

$$
\|e-e_0\|_\infty\le\varepsilon
\quad\Longrightarrow\quad B\le\varepsilon.
$$

Hệ số 1 phụ thuộc giả thiết chỉ risk E bị thay đổi, không áp tùy ý khi P và B cũng bị xấp xỉ.

### 10.5. Ghép cận B và A1

Khi cả hai cận cùng hợp lệ:

$$
\boxed{0\le\Delta\le\min\left\{
B,\ R_{A1}-R_S^*
\right\}.}
$$

Nếu không biện minh được B, vẫn có thể dùng cận A1 trong điều kiện tương ứng; không tự báo min như một bảo đảm.

## 11. Cận theo sai lệch lỗi tầng 2 và giới hạn áp dụng

### 11.1. Từ sai lệch miss rate và false-block rate

Giữ phí gọi c và chọn surrogate từ các rate hằng $m_0,f_0$:

$$
e_0(s)=c+As\,m_0+D(1-s)f_0.
$$

Khi đó:

$$
\delta(s)=As\big(m(s)-m_0\big)
+D(1-s)\big(f(s)-f_0\big).
$$

Bất đẳng thức tam giác cho:

$$
B\le\mathbb E\left[
AS|m(S)-m_0|+D(1-S)|f(S)-f_0|
\right].
$$

Nếu $|m(s)-m_0|\le\varepsilon_m$ và $|f(s)-f_0|\le\varepsilon_f$ trên miền cần xét, dùng tính hiệu chuẩn $\mathbb E S=\mathbb P(Y=1)$:

$$
\boxed{B\le
A\mathbb P(Y=1)\varepsilon_m
+D\mathbb P(Y=0)\varepsilon_f.}
$$

Cùng một sai lệch rate có ảnh hưởng khác nhau tùy chi phí và tỷ lệ lớp. Đây là lý do phải giữ riêng hai loại lỗi thay vì gộp bằng accuracy trung bình.

### 11.2. Những suy rộng chưa được phép

- Cận xét các giá trị tối ưu tổng thể trong lớp chính sách; không tự chặn chênh lệch test của hai chính sách học từ validation.
- Nếu chỉ cho phép ngưỡng trên biên bin, phải kiểm surrogate khả thi trong lớp bị giới hạn hoặc cộng thêm sai số xấp xỉ.
- B ước lượng từ mẫu không tự là cận tin cậy hữu hạn mẫu; cần kiểm soát sai số ước lượng nếu muốn phát biểu xác suất.
- B có đơn vị chi phí tuyệt đối trên giao dịch. Chỉ chia để ra regret tương đối khi risk mẫu số dương. Không có cận phần trăm hữu hạn chung khi risk tham chiếu bằng 0.
- Thêm ngân sách E hoặc thay phạm vi thông tin có thể thay đổi bài toán tối ưu và làm mất giả thiết của chứng minh.
- Kiểm thử số hỗ trợ phát hiện lỗi triển khai; không thay thế chứng minh.

Trên cùng bảng chi phí, lớp per-bin chứa lớp hai ngưỡng trên biên bin nên risk tối ưu per-bin không lớn hơn risk hai ngưỡng đó. Trên test, hai chính sách đã học có thể cho chênh lệch âm do nhiễu ước lượng; điều này không mâu thuẫn với $\Delta\ge0$ của tối ưu tổng thể.

## 12. Cơ sở lý thuyết của kiểm thử Chow T1

### 12.1. Ba yêu cầu bắt buộc

| Mục | Đối tượng | Tiêu chí |
|---|---|---|
| T1(a) | B1 Pietraszek với $a_+=a_-=c$ | Cut hợp lệ, theo thứ tự, cho risk Bayes trong mô hình Chow |
| T1(b) | B2 tìm hai ngưỡng tối ưu trên cùng risk | Risk bằng đáp án quét cặp cut độc lập |
| T1(c) | Oracle tổng thể theo score và oracle hai ngưỡng | $R_{2T}^*-R_S^*=0$ trong từng fixture Chow; B1/B2 cũng đạt risk oracle |

Các test tương ứng nằm trong `TestChowBaselines`: `test_t1a_b1_pietraszek_constant_abstention`, `test_t1b_b2_grid_same_risk` và `test_t1c_independent_oracle_regret_zero`.

### 12.2. Vì sao dùng fixture hiệu chuẩn chính xác?

Ở mỗi score $s=k/d$, tạo đúng d nhãn, trong đó có k nhãn dương. Phân phối score hữu hạn được chọn đều trên các điểm hỗ trợ. Khi đó xác suất lớp dương ở mỗi điểm được biết chính xác; không cần lấy mẫu Bernoulli để ước lượng risk.

Cách này kiểm công thức và thuật toán, tránh để lỗi lý thuyết bị che bởi nhiễu ngẫu nhiên hoặc để test thất bại do biến thiên lấy mẫu.

| Fixture | $(A,D,c)$ | $d$ và tử số $k$ của hỗ trợ score |
|---|---|---|
| `dyadic-ties` | $(8,2,0.5)$ | $d=16$; $k\in\{0,1,2,11,12,13,16\}$ |
| `project-100-1-c0.2` | $(100,1,0.2)$ | $d=1000$; $k\in\{0,1,2,3,790,800,810,1000\}$ |
| `near-critical-c0.99` | $(100,1,0.99)$ | $d=20000$; $k\in\{0,196,198,199,200,201,20000\}$ |
| `empty-esc-c1` | $(100,1,1)$ | $d=101$; $k\in\{0,1,2,101\}$ |

Bộ gần tới hạn có score 0.00995 trong khoảng E hẹp. Bộ c=1 kiểm trường hợp E rỗng cho đủ cả ba mục T1, thay vì chỉ kiểm công thức đóng.

### 12.3. Oracle phải độc lập

Oracle $R_S^*$ được tính trực tiếp bằng trung bình bao dưới:

$$
R_S^*=\frac1K\sum_{i=1}^K
\min\{As_i,c,D(1-s_i)\}.
$$

Oracle hai ngưỡng liệt kê mọi cặp cut có thứ tự, gồm các cut ngoại biên, rồi tính tổng risk bằng luật so sánh score trực tiếp. Nó không gọi các helper chọn action hoặc tính cost của thuật toán đang kiểm.

Nếu lấy output B2 làm expected của chính B2, cùng một lỗi có thể nằm ở cả hai phía phép so sánh. Tính độc lập của oracle là yêu cầu về thiết kế kiểm thử, không chỉ về việc dùng một tên hàm khác.

### 12.4. Nhiều cặp cut tương đương ở điểm hòa

Trong fixture dyadic:

$$
t_L^{\mathrm{Chow}}=\frac1{16},\qquad
 t_H^{\mathrm{Chow}}=\frac34.
$$

Do hỗ trợ score rời rạc và hòa ở hai biên, bốn cặp sau có cùng risk tối ưu:

$$
t_L\in\left\{\frac1{16},\frac18\right\},\qquad
 t_H\in\left\{\frac34,\frac{13}{16}\right\}.
$$

Test B1 chấp nhận các lựa chọn này và kiểm risk từng điểm bằng risk Bayes. Nó không khóa một cặp theo thứ tự duyệt hiện tại của solver. Đây là cách cho phép thay đổi ưu tiên phá hòa mà vẫn phát hiện nghiệm có risk sai.

### 12.5. Các ca biên và test cận

T1 còn kiểm ma trận chi phí trên bốn tổ hợp y,b; case affine theo lớp; điểm ngay trước/đúng/sau ngưỡng; ba hành động hòa; phí E quá đắt hoặc bằng 0; các cấu hình vùng rỗng; nghiệm toàn P/E/B; score trùng; hoán vị hàng; không sửa mảng đầu vào; một số tham số không hợp lệ.

`TestBoundA1AndB` kiểm riêng cận A1/B bằng kỳ vọng hữu hạn có trọng số không đều. Vì exact solver trả trung bình hàng, test nhân hàng i với $n w_i$ để thu được kỳ vọng $\sum_i w_i r_i$. Có test riêng cho surrogate E=50 tạo ngưỡng trùng trong khi risk E thật bằng 0.2, xác minh chuỗi cận vẫn đúng ở trường hợp suy biến.

Tên lớp test cận không dùng chữ T3 để tránh nhầm với phản ví dụ nhiều khoảng của Thuỳ.

## 13. Tiêu chí rà soát T3 của Thuỳ

T3 theo phân công là `tests/test_fragmented_regret.py`, kiểm phản ví dụ nhiều khoảng. Nó không phải test có chữ số 3 trong `test_roc.py` và không được thay thế bằng các test hỗ trợ cận ở trên.

Checklist lý thuyết và kiểm chứng:

- [ ] Fixture có ít nhất hai khúc ESC tách nhau bởi một vùng không ESC; ghi action sequence và biên các khúc.
- [ ] Ghi rõ phân phối S, trọng số có tổng 1, không gian rời rạc hay liên tục.
- [ ] Tính $R_S^*$ và $R_{2T}^*$ độc lập từ fixture, có đáp án cụ thể; không dùng output thuật toán đang kiểm làm expected.
- [ ] Kiểm $\Delta>0$ và khớp đáp án trong sai số có giải thích.
- [ ] Kiểm $\Delta\le B+\varepsilon$ khi giả thiết B được thỏa mãn; đồng thời kiểm cận A1 hoặc min hai cận.
- [ ] Cho phép E rỗng trong họ ngưỡng; score trùng không bị tách; hòa được quy định hoặc đối chiếu qua risk tối ưu.
- [ ] Nếu dùng tích phân số, tách sai số tích phân khỏi tolerance dấu phẩy động.

Một phản ví dụ có regret dương chưa tự kiểm được cận B, tính độc lập oracle hoặc tính đúng của trọng số. Các mục này phải được nghiệm thu riêng. Trong phiên bản tài liệu nguồn được tổng hợp, file T3 của Thuỳ chưa có trong checkout nên chưa xác nhận đạt checklist.

## 14. Sai số số học và tính nguyên vẹn dữ liệu

### 14.1. Sai số cho phép

Các test risk/ngưỡng giải tích dùng float64 với quy mô chi phí đã chọn:

```python
ATOL = 1e-12
RTOL = 1e-12
```

Điều kiện `allclose` là:

$$
|x-\hat x|\le\mathrm{ATOL}+\mathrm{RTOL}|\hat x|.
$$

ATOL giúp so gần 0; RTOL điều chỉnh theo độ lớn tham chiếu. Một số assertion dùng so chính xác hoặc `assertAlmostEqual` có chủ đích. Action, nhãn và bất biến mảng không được so xấp xỉ tùy ý.

`np.nextafter` tạo số biểu diễn được ngay trước/sau ngưỡng để kiểm dấu `<` và `>=`. Không dùng `isclose` trong luật dự đoán để biến một điểm hòa thành cả dải score. Chênh risk ở mức một ULP không phải lúc nào cũng phân biệt được bằng số học.

Tolerance trên phục vụ các fixture hiện tại; không phải chặn sai số thống kê hoặc bảo đảm chung cho mọi scale chi phí. Khi đổi scale mạnh, gần suy biến, dùng tích phân hoặc score ước lượng, cần phân tích lại sai số tương ứng.

### 14.2. Train, validation và test

Theo đặc tả đã chốt:

1. Scaler, mô hình và calibrator chỉ fit từ train.
2. Platt dùng dự đoán ngoài fold từ CV nội bộ train.
3. Validation dùng để chọn ngưỡng/chính sách.
4. Test dùng để đánh giá cuối.

Fit calibration trên val là sai quy trình đã yêu cầu, không được mô tả thành lựa chọn được chấp nhận. Test unit thành công không chứng nhận toàn pipeline tuân thủ các điều kiện này.

### 14.3. Gom bản trùng: chuẩn hóa biểu diễn số

“Chuẩn hóa” để so bản trùng nghĩa là parse và biểu diễn số nhất quán, chẳng hạn so vector đặc trưng theo giá trị float64 thay vì chuỗi CSV. Các chuỗi `1`, `1.0` có thể biểu diễn cùng giá trị.

Khóa nhóm trùng không cần fit tham số hoặc scaling; do đó bản thân thao tác chuẩn hóa biểu diễn này không tạo rò rỉ do học thống kê. Scaler của mô hình là bước khác và vẫn chỉ fit train.

Phải giữ hàng và gom các vector đặc trưng trùng vào cùng một split. Không dùng nhãn trong khóa nhóm; không tự làm tròn để gom cả các giá trị gần nhau nếu chưa có quy định. Xác thực NaN/Inf trước khi lập khóa.

Manifest cần ghi số hàng, số nhóm trùng, số hàng thuộc nhóm trùng, số bản dư $\sum_g(n_g-1)$ và việc nhóm được phân vào các split. Credit Card Fraud không có ID giao dịch để dùng thay cho khóa vector đặc trưng theo yêu cầu này.

### 14.4. Bất biến dữ liệu cần giữ

Score/bảng chi phí đầu vào không bị solver sửa ngoài ý muốn; hoán vị hàng không làm đổi risk tối ưu; score trùng nhận cùng action; độ dài mảng khớp; nhãn và quyết định tầng 2 thuộc miền hợp lệ; không có NaN/Inf trong các đầu vào yêu cầu hữu hạn.

Các kiểm tra này bảo vệ phép đo risk có cùng đối tượng. Chúng không thay thế kiểm split theo nhóm hoặc xác minh toàn bộ dữ liệu thực.

## 15. Quan hệ với mã nguồn, kiểm chứng và phần còn mở

### 15.1. Những sửa lỗi baseline liên quan trực tiếp lý thuyết

| Hiện tượng ở code gốc | Sửa để khớp mô hình |
|---|---|
| Quy đổi cut (0,0) tạo E thay vì toàn BLOCK | Dùng ánh xạ cut 0/K sang −∞/+∞ cho cả hai biên |
| Nghiệm all-PASS làm truy cập phần tử sa[n] | Trả +∞ khi chỉ số ngưỡng dưới bằng n |
| E quá đắt vẫn trả hai giao điểm đảo thứ tự | Kiểm điều kiện và trả $(t_0,t_0)$, đồng thời kiểm miền chi phí |

Các sửa nguồn trên nằm trong commit `02da4c207f1e6dbb5f7edca6de3bc85f5779a637`, được giữ trong commit đối chiếu mới `14bd0342a7d348ee0f23cdbcf806c82815bfbca6`. Bảng này giải thích quan hệ lý thuyết–triển khai, không có nghĩa báo cáo hiện tại sửa thêm code.

### 15.2. Kết quả kiểm chứng đã ghi nhận

Báo cáo nghiệm thu ngày 03/11/2026 ghi kết quả trên commit đối chiếu:

| Kiểm tra | Kết quả lịch sử có log |
|---|---|
| `python -m pip check` | Không có xung đột phụ thuộc được báo |
| `python -m pytest tests/test_chow_constant.py -q` | 25 passed |
| `python -m pytest tests/ -q` | 48 passed |

Môi trường nghiệm thu là Linux/CPython 3.12, với phụ thuộc trực tiếp và bắc cầu được ghim phiên bản trong `requirements.txt`, `requirements-dev.txt`. Các lớp unittest cũ vẫn được pytest chạy; fixture tham số mới yêu cầu pytest.

Lượt tổng hợp `theory.md` chỉ biên tập lý thuyết từ các nguồn đã đọc, không chạy lại test và không tạo kết quả thực nghiệm mới. Các kết quả trên không xác nhận T3 chưa có, chất lượng dữ liệu thực hoặc định lý tối ưu khoảng tổng quát.

### 15.3. Đối chiếu chứng minh gốc của nhóm

Chứng minh B trong báo cáo đã được trình bày với giả thiết rõ. Tuy nhiên, chưa nhận được bản báo cáo tuần 2/đề cương của Thuỳ và Phước để đối chiếu từng bước. Không coi việc chưa thấy trong code là bằng chứng nhóm chưa có chứng minh.

| Điểm cần rà ở bản nhóm | Phát biểu đang dùng trong báo cáo |
|---|---|
| Phạm vi thông tin | Mọi chính sách chỉ dùng S |
| Risk bị xấp xỉ | Chỉ E; P và B giữ nguyên |
| Surrogate khả thi | Bayes hai ngưỡng, kể cả $(t_0,t_0)$ |
| Phân phối và tích phân | Cùng phân phối S, risk khả tích |
| Ràng buộc bổ sung | Không ngân sách E ghép các quyết định |
| Cận sơ bộ | A1 thuộc lớp hai ngưỡng nhờ cho E rỗng |
| Ý nghĩa regret | Risk tối ưu tổng thể theo score |
| Đơn vị cận | Chi phí tuyệt đối/giao dịch |

Các hàng này hiện là nội dung chờ đối chiếu với bản gốc, không phải xác nhận nhóm đã dùng cùng giả thiết.

### 15.4. Các việc chưa hoàn tất trong phạm vi nguồn

- Chưa rà fixture và đáp án T3 của Thuỳ vì chưa có file tương ứng.
- Chưa nhận bản chứng minh B của nhóm để hoàn tất rà soát bản gốc.
- Chưa sửa các lỗi pipeline dữ liệu đã báo: Platt fit val và bỏ dòng trùng trong bản code đối chiếu; fixture cũ của `test_roc.py` cũng cần đồng bộ quy trình.
- Chưa chạy đánh giá Credit Card thực trong phần việc đã bàn giao.
- Chưa chứng minh điều kiện cần và đủ tổng quát cho tính tối ưu khoảng khi tầng 2 phụ thuộc score.

## 16. Kết luận và nguồn tổng hợp

### 16.1. Những điểm lý thuyết đã chốt

Score có hướng tăng theo nguy cơ lớp 1; công thức Chow dùng giả thiết hiệu chuẩn. Chi phí chuyển tầng thực tế cộng phí gọi và lỗi cuối của tầng 2. Nghiệm Chow bất đối xứng được suy từ bao dưới ba risk, có kiểm điều kiện vùng E tồn tại và có nghiệm một ngưỡng khi E rỗng.

Chi phí hằng theo lớp vẫn cho một nghiệm Bayes thuộc lớp hai ngưỡng, kể cả suy biến. Khi risk E phụ thuộc score, khả năng nhiều khoảng phải được kiểm bằng cấu trúc risk thay vì chỉ dựa vào ROC tầng 1. Regret hạn chế hai ngưỡng có cận A1 và, dưới giả thiết chỉ risk E thay đổi, có cận B bằng kỳ vọng độ lệch tuyệt đối của risk E.

T1 kiểm công thức và các baseline bằng fixture hiệu chuẩn, oracle độc lập và tiêu chí hòa không phụ thuộc một cặp cut đồng hạng. T3 phản ví dụ nhiều khoảng và việc đối chiếu chứng minh của nhóm có checklist riêng, không được suy ra là hoàn tất từ test T1 thành công.

### 16.2. Nguồn nội bộ đã đọc để tổng hợp

1. [test_spec.md — phiên bản 2](https://chatgpt.com/library?file_id=libfile_afadfb3517408191b5e0a88276770161&account_id=personal): quy ước, công thức, T1, checklist T3 và dữ liệu.
2. [bound_B.md — phiên bản 1](https://chatgpt.com/library?file_id=libfile_a35c7ab20bec8191b63fc96dda8ecdd6&account_id=personal): cận B, liên hệ A1 và phần chờ đối chiếu.
3. [validation_report.md — phiên bản 1](https://chatgpt.com/library?file_id=libfile_b145c45bbd3081919933a28f3cdebafb&account_id=personal): SHA, kết quả lịch sử và giới hạn nghiệm thu.

Báo cáo này tự chứa các định nghĩa, dẫn xuất, chứng minh và lý do lựa chọn. Các liên kết nguồn phục vụ truy vết phiên bản; không cần mở các nguồn để đọc phần lý thuyết chính.
