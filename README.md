# Đồ án Khoa học Dữ liệu

Pipeline: **EDA → Preprocessing → Modelling**. Hiện repo có framework EDA (univariate + bivariate) cho bài toán phân loại nhị phân.

## 1. Cấu trúc repo

```
.
├── README.md
├── requirements.txt
├── data/                          # dữ liệu thô (không đẩy lên git)
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb   # Giai đoạn 3: Preprocessing
└── src/
    └── eda/
        ├── univariate.py          # UnivariateAnalyzer: phân tích từng cột liên tục/đếm
        └── bivariate.py           # BivariateAnalyzer: feature vs target, feature vs feature
```



## 2. Cách xem chi tiết một hoặc nhiều cột: `selected_cols`

Ở các cell phân tích chi tiết, bạn chỉ cần **sửa danh sách `selected_cols`** rồi chạy lại cell.

```python
# Ví dụ: muốn xem chi tiết 2 cột này
selected_cols = ["domain_age", "length_url"]
```

- Tên cột phải **đúng chính tả, phân biệt hoa/thường** và có trong dữ liệu.
- Xem 1 cột: `selected_cols = ["domain_age"]` (vẫn phải để trong dấu `[ ]`).
- Xem tất cả cột liên tục: `selected_cols = cont_cols`.
- Xem nhanh các cột "đáng chú ý" thay vì tự gõ tên:

```python
selected_cols = ua.top("skew", n=8)                 # 8 cột lệch nhất
selected_cols = ua.top("pct_out", n=8)              # 8 cột nhiều outlier nhất
selected_cols = ua.filter_by_flag("HIGH_SKEW")      # mọi cột lệch mạnh
selected_cols = ba.top_target(8)                    # 8 cột tách lớp tốt nhất
```

### 2.1 Univariate: xem chi tiết cột

```python
selected_cols = ["domain_age", "length_url"]

ua.report(selected_cols)                        # bảng ngang: Validity / Descriptive / Distribution / Outlier
ua.report(selected_cols, layout="list")         # dạng danh sách dọc
ua.show(selected_cols, ncols=2)                 # thẻ báo cáo xếp ngang
ua.plot(selected_cols, ncols=4)                 # Hist+KDE + Boxplot
ua.plot(selected_cols, ncols=4, clip_upper_q=0.99)   # cắt đuôi 1% để thấy rõ thân phân phối
```

- Cột có đuôi quá dài, biểu đồ bị dồn sát một bên: thêm `clip_upper_q=0.99`.
- Chỉ cần bảng tổng quan toàn bộ cột: `ua.summary()` (không cần `selected_cols`).

### 2.2 Bivariate: Feature vs Target

```python
selected_cols = ["domain_age", "length_url"]

ba.target_table(selected_cols)                  # AUC, sep, KS, spearman... của các cột đã chọn
ba.plot_target(selected_cols, ncols=4)          # KDE + Boxplot theo từng lớp
```

Cột nhị phân (0/1):

```python
selected_bin_cols = ["has_ip", "https"]
ba.binary_target_table(selected_bin_cols)
ba.plot_binary_target(selected_bin_cols)
```

Xếp hạng toàn bộ (không cần chọn cột): `ba.plot_target_rank()`.

### 2.3 Bivariate: Feature vs Feature

```python
selected_cols = ["length_url", "length_hostname", "nb_dots"]

ba.corr_pairs(selected_cols, thr=0.8)           # các cặp tương quan cao trong nhóm đã chọn
ba.plot_corr(selected_cols)                     # heatmap tương quan

selected_pairs = [("length_url", "length_hostname")]
ba.plot_scatter(selected_pairs)                 # scatter từng cặp, tô màu theo target
```

Quét toàn bộ dữ liệu: `ba.redundant_groups(thr=0.8)` (không cần `selected_cols`).

## 4. Cách đọc kết quả

**Cờ cảnh báo của Univariate (cột `flags`)**

| Flag | Ý nghĩa |
|---|---|
| `HAS_NAN` / `INF_VALUE` | Có giá trị thiếu / vô cực |
| `NEGATIVE_VALUE` | Có giá trị âm (kiểm tra có hợp lệ không) |
| `CONSTANT` | Chỉ có 1 giá trị, nên loại |
| `LOW_CARDINALITY` | ≤ 10 giá trị unique, coi như biến rời rạc |
| `ZERO_INFLATED` | ≥ 50% giá trị bằng 0 |
| `HIGH_SKEW` / `MODERATE_SKEW` | Lệch mạnh (|skew| ≥ 1) / lệch vừa (≥ 0.5) |
| `MANY_OUTLIERS` | > 5% outlier theo IQR |

**Chỉ số của Bivariate**

| Chỉ số | Cách đọc |
|---|---|
| `sep = \|2*AUC-1\|` | ≥ 0.5 tách lớp mạnh; < 0.1 gần như không tách lớp |
| `dir` | `+`: lớp 1 có giá trị cao hơn; `-`: thấp hơn |
| `KS` | Khác biệt lớn nhất giữa 2 phân phối (càng lớn càng khác) |
| `phi` (nhị phân) | Tương quan có dấu với target |
| `corr_pairs` | Cặp |corr| ≥ `thr` có thể dư thừa, giữ cột có `sep` cao hơn |

## 5. Quy trình gợi ý

1. Chạy các cell khởi tạo (import, đọc dữ liệu, `cont_cols`, `bin_cols`, tạo `ua`, `ba`).
2. `ua.summary()` để có cái nhìn tổng thể, rồi chọn các cột có cờ đáng chú ý.
3. Gán các cột đó vào `selected_cols`, chạy `ua.report` + `ua.plot` để xem chi tiết.
4. Chạy phần Bivariate: xếp hạng cột theo `sep`, tìm cột `WEAK_SEP` và các nhóm dư thừa.
5. Tổng hợp thành bảng đề xuất xử lý cho từng cột (thiếu giá trị, outlier, lệch, giữ hay bỏ) để dùng ở Giai đoạn Preprocessing.

## 6. Lỗi thường gặp

| Lỗi | Cách sửa |
|---|---|
| `KeyError: Cột không tồn tại` | Tên cột trong `selected_cols` sai chính tả |
| `ModuleNotFoundError: src` | Kiểm tra `sys.path.append("..")` ở cell đầu, hoặc tạo `src/__init__.py` và `src/eda/__init__.py` |
| Lỗi `hue`/`legend` khi vẽ boxplot | `pip install -U seaborn` (cần >= 0.13) |
| Biểu đồ dồn sát một bên | Thêm `clip_upper_q=0.99` |
