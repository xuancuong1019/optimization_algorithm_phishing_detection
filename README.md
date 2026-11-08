# Hướng dẫn đọc notebook `01_exploratory_data_analysis`

Notebook này là bước EDA cho bài toán phân loại URL **phishing (1) / legitimate (0)** trên bộ dữ liệu `dataset_B_05_2020.csv` (11.430 URL, 87 đặc trưng). Phần này giải thích cách đọc từng bảng và biểu đồ để người nhận bàn giao không phải đoán.

## 0. Đọc nhanh

**Thứ tự nên đọc:** Mục 1 (tóm tắt, bảng 1.2) → các khung "Đánh giá" dưới mỗi mục → hai file CSV trong `reports/` → xem lại biểu đồ khi cần.

**Quy ước chung cho mọi biểu đồ**
- **Xanh (`#4C72B0`)** = giá trị 0 / legitimate. **Cam (`#DD8452`)** = giá trị 1 / phishing.
- Nhãn cân bằng 50/50 nên **đường nét đứt 50%** là mốc "không phân biệt được"; càng xa 50% tín hiệu càng mạnh.
- Cột chữ bên phải biểu đồ thường là **số dòng (`n`)**. Luôn đọc `n` cùng với tỷ lệ phần trăm.
- Thang `log` ở trục tần suất để nhìn được cả cột lớn lẫn cột chỉ vài chục dòng.

**Thuật ngữ**

| Thuật ngữ | Ý nghĩa | Cách đọc |
| :--- | :--- | :--- |
| `sep` | `\|2·AUC − 1\|`, độ tách hai lớp của một cột số | 0 = không tách được, 1 = tách hoàn toàn. Ngưỡng dùng: `≥ 0,35` mạnh, `< 0,10` yếu |
| `AUC` | Xác suất một dòng phishing có giá trị cao hơn một dòng legit | `> 0,5` giá trị cao nghiêng phishing, `< 0,5` nghiêng legit |
| `ρ` / `spearman` | Tương quan hạng với nhãn (hoặc giữa 2 cột) | Dấu `+` / `−` cho biết chiều |
| `KS` | Khoảng cách lớn nhất giữa 2 phân phối | Càng lớn, hai lớp càng khác nhau |
| `skew` | Độ lệch của phân phối | `≥ 1` lệch phải mạnh (đuôi dài bên phải) |
| `max\|z\|` | Giá trị xa trung bình nhất, tính bằng số độ lệch chuẩn | Lớn (> 10) tức là có ngoại lai cực đoan |
| `\|r\|max` | Max của `\|Pearson\|` và `\|Spearman\|` giữa 2 cột | Dùng ở Mục 5.2 để không sót cặp nào |

**Cảnh báo chung:** mọi con số được tính trên **toàn bộ dữ liệu**, chưa chia train/test. Dùng để hiểu dữ liệu và gợi ý, không dùng để chọn đặc trưng cuối cùng.

## 1. Tổng quan dữ liệu

| Hình / bảng | Cho biết gì | Cách đọc | Cần chú ý |
| :--- | :--- | :--- | :--- |
| Bảng 1.1 | 87 cột chia thành 5 nhóm (IU1, IU2, IC1, IC2, E) | Mỗi dòng là một nhóm, cột "Cách lấy" cho biết cột đến từ đâu | Nhóm IC1/IC2 cần tải được trang HTML; nhóm E cần dịch vụ ngoài |
| Bảng 1.2 | Những vấn đề dữ liệu ảnh hưởng đến bước xử lý sau | Mỗi dòng: vấn đề, cột liên quan, số liệu, quyết định cần đưa ra | **Đọc kỹ đầu tiên.** Dòng 5 và 6 nói về cột có nghĩa khác tên |
| Ô 1.4 (in ra màn hình) | Kích thước, NaN, dòng trùng, phân phối nhãn, nhóm cột, giá trị âm, cặp tương quan cao | Đọc từ trên xuống, mỗi `print` là một kiểm tra | Cặp tương quan ở ô này chỉ dùng Pearson, nên thiếu một số cặp (xem 5.2) |

## 3. Phân tích đơn biến

### 3.1. Phân loại cột (`col_info`, xuất ra `reports/column_types.csv`)

| Cột trong bảng | Ý nghĩa |
| :--- | :--- |
| `group`, `group_name` | Nhóm theo bài báo gốc |
| `type`, `type_key` | Loại cột do notebook tự phân (binary, near-constant, few-levels, ...) |
| `dict_type`, `dict_vs_data` | Kiểu dictionary ghi, và dữ liệu thực tế có khớp không (`Khớp`, `LỆCH`, `hằng số`) |
| `ghi_chu` | Ghi chú tự sinh, kèm mức chắc chắn: `[xác nhận]`, `[suy luận]`, `[giả thuyết]`, `[chưa xác nhận]` |

**Cách đọc:** tra theo tên cột. `[xác nhận]` có thể tin; `[suy luận]` và `[giả thuyết]` cần đối chiếu nguồn trước khi dùng.

### 3.2. Cột nhị phân (31 cột)

| Hình | Cho biết gì | Cách đọc | Cần chú ý |
| :--- | :--- | :--- | :--- |
| Thanh xếp chồng "Tỷ lệ giá trị 0/1" | Mỗi thanh là một cột nhị phân; xanh = % bằng 0, cam = % bằng 1 | Sắp theo % giá trị 1 giảm dần từ trên xuống. Cột phải ghi `% (số dòng)` của giá trị 1 | Hai đường nét đứt ở `5%` và `95%`: cột vượt đường là **gần hằng số** (giá trị 1 quá hiếm). 16/31 cột rơi vào vùng này |

### 3.4. Cột áp đảo một giá trị và cột hằng số (15 cột)

| Hình / bảng | Cho biết gì | Cách đọc | Cần chú ý |
| :--- | :--- | :--- | :--- |
| Bảng 1 | 6 cột hằng số và 9 cột bị một giá trị áp đảo | `giá trị chủ đạo`, `n khác chủ đạo`, `% khác chủ đạo`, `số mức`, `max\|z\|` | Hằng số có `n khác chủ đạo = 0`. Cột áp đảo thường có `max\|z\|` rất lớn |
| Thanh ngang (trục log) | Số dòng khác giá trị chủ đạo của 9 cột áp đảo | Càng dài càng có nhiều tín hiệu thật | Đường nét đứt ở **100 dòng**; cột dưới đường quá thưa |
| Bảng 2 (`url_consistency`) | Cột có khớp công thức đếm trực tiếp trên chuỗi URL không | `% dòng khớp` | **Gần 100%** nghĩa là định nghĩa đúng và giá trị khác chủ đạo là dữ liệu thật, không phải lỗi trích xuất |

### 3.5. Cột đếm ít mức, không bị áp đảo (7 cột)

| Hình / bảng | Cho biết gì | Cách đọc | Cần chú ý |
| :--- | :--- | :--- | :--- |
| Bảng 1 | Khoảng giá trị, số mức, mức chủ đạo, các mức hiếm | `mức hiếm (<100 dòng)` liệt kê mức ít dòng | Mức hiếm là ứng viên để gộp đuôi (`clip`) |
| Bảng 2 | Độ lệch và ngoại lai | So `skew` với `skew_log1p`: `log1p` có giúp không | Nếu `skew_log1p` vẫn cao thì `log1p` chưa đủ, nên dùng `clip` |
| Lưới 7 biểu đồ cột (trục log) | Tần suất từng mức của từng cột | Cột **xanh** = mức có ≥ 100 dòng (ghi %); cột **xám** = mức hiếm (ghi số dòng) | Mức xám nằm giữa các mức xanh (như `page_rank` mức 9) là bình thường, không phải ngoại lai |
| Bảng 3 (`zero_sentinel_check`) | `page_rank = 0` có phải là "không có dữ liệu" không | So `%page_rank=0 khi đúng` với `khi sai` | Nếu hai số gần nhau thì số 0 không hẳn là báo lỗi |

### 3.6. Cột đếm nhiều mức và tỷ lệ, đa số bằng 0 (10 cột)

| Hình / bảng | Cho biết gì | Cách đọc | Cần chú ý |
| :--- | :--- | :--- | :--- |
| Bảng 1 | Thống kê chỉ trên phần `x > 0` | `% bằng 0`, `median`, `p95`, `max`, `skew` | Cột `% bằng 0 (trang tải được)` cho biết bao nhiêu số 0 là **thật** sau khi bỏ trang chết |
| Bảng 2 | Độ lệch và `max\|z\|` trên **toàn bộ** cột (cả số 0) | So với Bảng 1: toàn cột thường lệch mạnh hơn | Cột có > 85% số 0 thì `log1p` trực tiếp không đủ |
| Lưới 10 biểu đồ | Phân phối phần `x > 0`: cột đếm vẽ dạng cột, cột tỷ lệ vẽ dạng histogram | Tiêu đề phụ ghi `% bằng 0` và số dòng `x > 0`. Nhãn `14+` nghĩa là "14 trở lên" (đã gộp đuôi) | Cột gai ở biên phải (`1.0` hoặc `100`) là nhiều site dùng 100% tài nguyên nội bộ/ngoài |

### 3.7. Cột liên tục thực (24 cột)

| Hình / bảng | Cho biết gì | Cách đọc | Cần chú ý |
| :--- | :--- | :--- | :--- |
| Bảng `continuous_matrix` | Rủi ro scale/biến đổi của từng cột, sắp theo `skew` giảm dần | Cột `cờ` là chẩn đoán: `ÂM`, `NHIỀU 0`, `DẢI RỘNG`, `THANG 0–100`, `LỆCH MẠNH`, `log1p ĐỦ` / `CHƯA ĐỦ` / `LỆCH NGƯỢC` | `log1p LỆCH NGƯỢC` (như `domain_age`) nghĩa là **không** dùng `log1p` |
| Lưới 4×6 histogram | Hình dạng phân phối từng cột | Màu: **đỏ** lệch mạnh, **vàng** lệch vừa, **xanh lá** gần đối xứng | Biểu đồ đã bỏ số âm, số 0 ở cột nhiều số 0, trang chết ở cột HTML, và cắt trục ở P99. Ghi chú xám dưới mỗi hình nói rõ đã bỏ gì |

## 4. Phân tích hai biến

### 4.1.1. Đặc trưng số và nhãn

| Hình / bảng | Cho biết gì | Cách đọc | Cần chú ý |
| :--- | :--- | :--- | :--- |
| a. Bảng `tb_num_biv` | Xếp hạng 50 cột số theo khả năng tách 2 lớp | `sep`, `AUC`, `chiều`, `spearman`, `KS`, `med_Legit` / `med_Phish` | Đã xử lý trước: số âm thành NaN ở 2 cột WHOIS, `ratio_extRedirection` cắt ở 1,0 |
| a. Thanh ngang xếp hạng | Độ dài thanh = `sep` | **Cam (+)**: giá trị lớn thiên về phishing; **xanh (−)**: giá trị lớn thiên về legit. Cột phải: loại cột và `%Phish (khi != mode)` | Hai đường nét đứt ở `0,10` và `0,35` |
| b. Thanh phishing/legit cho 19 cột thưa | Khi cột "có tín hiệu" (`x != mode`), bao nhiêu % là phishing | Cột phải: nhóm, `n` (và % dòng), `sep(x>0)` | Dòng ít `n` thì tỷ lệ kém ổn định. `sep(x>0)` cho biết độ lớn có giúp tách lớp thêm không |
| b. Lưới 7 biểu đồ cột cho cột đếm ít mức | % phishing theo từng bậc giá trị | Cột cam: ≥ 50% phishing; cột xanh: < 50%. Nhãn `n=` là số dòng của bậc | Đuôi đã được gộp (ví dụ `nb_www` gộp thành `1+`) để đỡ nhiễu |
| c. Lưới KDE + boxplot, 12 cột | Phân phối giá trị theo nhãn của 12 cột liên tục mạnh nhất | Trên: đường mật độ (xanh legit, cam phishing). Dưới: boxplot, vạch đen là trung vị. Tiêu đề ghi `sep` và `ρ` | Hai phân phối càng tách rời nhau càng tốt. Nhãn trục dưới cho biết dùng `log1p(x)` hay giá trị gốc cắt ở P99 |

### 4.1.2. Đặc trưng nhị phân và nhãn

| Hình | Cho biết gì | Cách đọc | Cần chú ý |
| :--- | :--- | :--- | :--- |
| Thanh xếp chồng "Phishing vs Legitimate given Binary Feature = 1" | Khi cột bằng 1, bao nhiêu % là phishing (cam) hay legit (xanh) | Sắp theo % phishing giảm dần. Cột phải: `n (x = 1)` | Cột `100%` phishing nhưng `n` nhỏ (như `punycode` 4 dòng) **chưa đủ để tin**. Cột có tên "dấu hiệu tốt" mà nghiêng phishing nhiều khả năng bị **đảo nghĩa** (xem 1.2, dòng 5) |

## 5. Quan hệ giữa các đặc trưng (5.2)

Phân tích trên 81 cột (đã bỏ 6 cột hằng số).

| Hình / bảng | Cho biết gì | Cách đọc | Cần chú ý |
| :--- | :--- | :--- | :--- |
| Bảng cặp (`pairs`, xuất `reports/feature_pairs_r50.csv`) | Các cặp cột tương quan mạnh | `spearman`, `pearson`, `\|r\|max`, `lệch P-S`, `ghi chú` | `lệch P-S` lớn (≥ 0,3) nghĩa là hai loại tương quan cho kết quả rất khác nhau, nên dùng cả hai. `ghi chú` cho biết quan hệ tất định đã biết |
| Heatmap 81×81 | Toàn cảnh tương quan sau khi gom cụm | Màu sáng = `\|r\|` cao. Màu chữ ở nhãn trục là nhóm cột | Tìm các **khối sáng dọc đường chéo**: đó là các cụm cột dư thừa |
| Ma trận 5×5 | `\|r\|` trung bình giữa các nhóm | Đường chéo = nội bộ nhóm; ngoài đường chéo = giữa hai nhóm | Nội bộ cao hơn liên nhóm là bình thường. Khối liên nhóm cao chủ yếu do hiệu ứng trang chết |
| Bảng cặp liên nhóm | Cặp cột khác nhóm nhưng tương quan cao | Như bảng cặp | Phần lớn đáng ngờ về ý nghĩa cột hơn là dư thừa thật (ví dụ `ip` với `ratio_digits_url`) |
| Bảng cụm (`reports/feature_clusters.csv`) | Các nhóm cột dư thừa và **đại diện gợi ý** | `các cột`, `\|r\| nội bộ TB`, `đại diện gợi ý`, `\|r\| nhãn từng cột` | Đại diện chọn theo `\|r\|` với nhãn trên toàn bộ dữ liệu, chỉ là **gợi ý**. Cần tính lại trên tập train |

## Lưu ý về cách đánh số
Notebook có các mục 1, 3, 4, 5.2, 6. Mục 2, 3.3 và 5.1 chưa có hoặc đã được gộp vào mục khác. Mục 6 (Đa biến) chưa làm.