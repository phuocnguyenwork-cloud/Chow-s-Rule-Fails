# 2 Cơ sở lý thuyết của chính sách ba hành động

Xét nhãn $Y=1$ cho giao dịch gian lận và $Y=0$ cho giao dịch hợp lệ. Tầng định tuyến quan sát điểm $s$ đã hiệu chuẩn, nghĩa là $s=P(Y=1\mid S=s)$. Ba hành động là cho qua $P$, chuyển lên bộ phân lớp tầng 2 $E$, và chặn $B$. Ký hiệu $A>0$ là chi phí bỏ sót gian lận, $D>0$ là chi phí chặn nhầm, $c\ge0$ là phí gọi tầng 2. Trong cấu hình thực nghiệm, $(A,D,c)=(100,1,0.2)$. Mọi so sánh chính sách dưới đây dùng cùng score, cùng phân phối và cùng đơn vị chi phí trên mỗi giao dịch.

## 2.1 Nghiệm Chow và trường hợp chi phí theo lớp

Nếu tầng 2 hoàn hảo, ba rủi ro điều kiện là $r_P(s)=As$, $r_E(s)=c$ và $r_B(s)=D(1-s)$. Giao điểm $P/E$ và $E/B$ lần lượt cho $t_L=c/A$ và $t_H=1-c/D$. Chính sách Bayes chọn $P$ khi $s<t_L$, $E$ khi $t_L\le s<t_H$, và $B$ khi $s\ge t_H$, miễn là $c<AD/(A+D)$. Với chi phí dự án, hai ngưỡng là $0.002$ và $0.8$. Nếu $c\ge AD/(A+D)$, vùng chuyển tầng rỗng; nghiệm là một ngưỡng $t_L=t_H=D/(A+D)$. Điểm bằng ngưỡng được gán sang hành động bên phải để tái lập kết quả; khi các rủi ro hòa nhau, mọi hành động có rủi ro nhỏ nhất đều là Bayes tối ưu.

Khi chi phí chuyển tầng là hằng theo từng lớp, đặt $a_+$ cho lớp gian lận và $a_-$ cho lớp hợp lệ. Rủi ro chuyển tầng là hàm affine $e_0(s)=a_+s+a_-(1-s)$. Hai giao điểm trở thành $t_L=a_-/(A-a_++a_-)$ và $t_H=(D-a_-)/(D-a_-+a_+)$. Vùng $E$ có độ dài dương đúng khi $a_+/A+a_-/D<1$. Nếu bất đẳng thức không đúng, chọn hai ngưỡng trùng tại $D/(A+D)$. Trường hợp Chow cổ điển thu được bằng cách đặt $a_+=a_-=c$. Baseline ROC của Pietraszek ước lượng $a_+,a_-$ từ lỗi tầng 2 theo lớp rồi chọn ngưỡng trên ROC tầng 1 theo mô hình affine này.

## 2.2 Khi tầng 2 có lỗi phụ thuộc score

Gọi $m(s)$ là xác suất tầng 2 bỏ sót gian lận và $f(s)$ là xác suất tầng 2 chặn nhầm giao dịch hợp lệ, có điều kiện theo score. Khi đó $r_E(s)=c+As\,m(s)+D(1-s)f(s)$. Chính sách Bayes theo score chọn hành động có rủi ro thấp nhất tại từng $s$. Nếu $m(s)$ hoặc $f(s)$ thay đổi mạnh, tập score mà $E$ rẻ nhất có thể gồm nhiều khoảng tách nhau. ROC của tầng 1 chỉ mô tả khả năng phân biệt nhãn của score; nó không xác định $m(s)$ và $f(s)$, nên không đủ để bảo đảm chính sách hai ngưỡng tối ưu cho cả hệ thống.

Fixture rời rạc năm bin của nhóm minh họa hiện tượng này. Chính sách theo từng bin chọn $P,E,B,E,B$ với risk tính tay $R_S^*=0.18576$. Chính sách hai ngưỡng tốt nhất có $R_{2T}^*=0.19149$, nên regret cấu trúc $\Delta=R_{2T}^*-R_S^*=0.00573>0$. Hai khúc chuyển tầng ở bin 1 và 3 bị vùng chặn ở bin 2 ngăn cách. Đây là một phản ví dụ hữu hạn, không phải chứng minh rằng mọi tầng 2 bất toàn đều làm hỏng cấu trúc hai ngưỡng.

## 2.3 Cận regret và phạm vi áp dụng

Cho $R_{A1}$ là risk của chính sách một ngưỡng chỉ dùng $P/B$ tốt nhất. Vì lớp hai ngưỡng cho phép $t_L=t_H$, ta có $R_S^*\le R_{2T}^*\le R_{A1}$ và $0\le\Delta\le R_{A1}-R_S^*$. Trên fixture, $R_{A1}=0.26450$, nên cận này bằng $0.07874$.

Một cận khác dùng surrogate affine $e_0(s)$, giả sử chỉ rủi ro của $E$ thay đổi còn $r_P,r_B$ giữ nguyên. Đặt $\delta(s)=r_E(s)-e_0(s)$ và chọn một chính sách Bayes của surrogate $\pi_0$ có dạng hai ngưỡng. Vì $\pi_0$ là chính sách khả thi, $\Delta\le R(\pi_0)-R_S^*$. Tại từng score, tính tối ưu của $\pi_0$ dưới surrogate cho thấy chênh rủi ro thật giữa $\pi_0$ và oracle không vượt $|\delta(s)|$: phần sai lệch chỉ xuất hiện khi một trong hai chính sách chọn $E$. Lấy kỳ vọng được $\Delta\le R(\pi_0)-R_S^*\le B$, với $B=\mathbb E|\delta(S)|$. Nếu cả hai cận hợp lệ, $\Delta\le\min\{B,R_{A1}-R_S^*\}$. Với surrogate có miss rate phẳng trong fixture, $R(\pi_0)-R_S^*=0.096985$ và $B=0.1275$; cận A1 chặt hơn trong ví dụ này.

Cận $B$ đòi hỏi cùng phân phối score, surrogate Bayes có nghiệm hai ngưỡng, rủi ro khả tích và không có ràng buộc ngân sách chuyển tầng ghép các giao dịch. Phản ví dụ cùng các cận trên chưa cho điều kiện cần và đủ tổng quát để một tầng 2 phụ thuộc score vẫn có nghiệm tối ưu hai ngưỡng; đó là phần lý thuyết nhóm tiếp tục nghiên cứu.
