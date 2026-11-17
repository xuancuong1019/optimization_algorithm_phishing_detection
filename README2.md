# Metaheuristic Feature Selection cho Phishing Website Detection

Khung (template) so sánh 4 thuật toán tối ưu metaheuristic (SA, GA, PSO, ACO) trong bài toán **chọn đặc trưng kiểu wrapper**, chạy trên 2 bộ dữ liệu:

| Tên trong config | Bộ dữ liệu | Mẫu / đặc trưng | File cần đặt vào `data/raw/` |
|---|---|---|---|
| `hannousse` | Web Page Phishing Detection (Hannousse & Yahiouche) | 11.430 / 87, nhãn chữ `status` | `dataset_phishing.csv` (đã kèm) |
| `vrbancic` | Datasets for phishing websites detection (Vrbančič et al., Mendeley Data 10.17632/72ptz43s9v.1) | 58.645 (`dataset_small`) hoặc 88.647 (`dataset_full`) / 111, nhãn 0/1 `phishing` | `dataset_small.csv` (**chưa kèm, cần tải và đặt vào**) |

`dataset_small` gần cân bằng (30.647 phishing / 27.998 legitimate); `dataset_full` mất cân bằng (30.647 / 58.000). Mặc định dùng `dataset_small`; muốn đổi sang bản full, sửa `raw_path` của `vrbancic` trong `config.py`. Lưu ý bộ này dùng giá trị `-1` cho thành phần URL không tồn tại (xử lý ở bước tiền xử lý, hiện chưa làm).

## Cấu trúc

```
data/
  raw/                 dữ liệu gốc
  processed/           dữ liệu sau tiền xử lý (tự tạo khi chạy)
src/
  config.py            toàn bộ cấu hình + các kịch bản
  preprocess.py        kiểm tra đầu vào, hàm làm sạch riêng từng dataset (khung), chia train/test, impute + scale
  optimizers.py        SA, GA, PSO, ACO (+ RS) và Tracker (cache, đếm, dừng, đo thời gian fitness)
  models.py            classifier: SVM, RF (dùng trong wrapper), LightGBM (chỉ đánh giá cuối)
  metrics.py           metric chọn cho fitness + công thức fitness
  pipeline.py          chạy MỘT (dataset, thuật toán, seed) và TRẢ VỀ kết quả thô, không in, không so sánh
  evaluate.py          đọc kết quả thô, in, so sánh, vẽ biểu đồ, xuất báo cáo
  main.py              điều phối: vòng lặp dataset x seed x thuật toán, lưu kết quả, gọi evaluate
results/               mỗi lần chạy: results/<kịch bản>_<thời gian>/
requirements.txt
README.md
```

## Cách chạy

```bash
pip install -r requirements.txt

python -m src.main --scenario quick                      # chạy thử nhanh (vài phút)
python -m src.main --scenario compare                    # so sánh đầy đủ: 2 dataset, 10 seed, wrapper SVM
python -m src.main --scenario compare_rf                 # wrapper dùng Random Forest
python -m src.main --scenario compare_with_lightgbm      # đánh giá cuối thêm LightGBM
python -m src.main --scenario compare --datasets hannousse         # chỉ chạy một dataset
python -m src.main --scenario compare --wrapper rf --extra-final lightgbm
python -m src.evaluate results/<thư mục kết quả>         # tạo lại báo cáo từ kết quả đã lưu, không chạy lại thí nghiệm
```

Chạy từ thư mục gốc. Các kịch bản khác (`compare_early_stop`, `compare_unique_budget`, `compare_no_cache`) xem `SCENARIOS` trong `config.py`.

Hạ `max_evals`, `seeds` hoặc `search_max_samples` nếu cần. SVM huấn luyện cuối trên phần train rất lớn của dataset 2 (~47.000 mẫu) có thể mất vài phút mỗi lần fit; đặt `final_max_samples` (vd 15000) để giới hạn, áp dụng như nhau cho baseline và mọi thuật toán.

## Luồng hoạt động (cho mỗi dataset, mỗi seed)

1. **Tiền xử lý và chia dữ liệu** (`preprocess.py`, stratified theo seed): 20% là **test** (không đụng tới khi tìm kiếm); 80% còn lại là **train**. CV fit imputer/scaler riêng trên từng fold-training. `pipeline.py` chọn một phần train làm **search set**: `min(search_ratio x |train|, search_max_samples)` mẫu, để giữ chi phí fitness chấp nhận được trên dataset lớn. Baseline và mọi thuật toán của cùng (dataset, seed) dùng chung đúng một bộ dữ liệu này (mảng đặt chỉ đọc; `evaluate` kiểm tra lại bằng mã băm dữ liệu).
2. **Vòng wrapper**: thuật toán sinh mask nhị phân -> huấn luyện classifier wrapper (`svm` hoặc `rf`, chọn bằng `wrapper_model`) qua **CV k-fold** trên search set (mặc định k = 3, fold cố định trong suốt lần chạy) -> fitness -> thuật toán cập nhật, đến khi dừng.
   `fitness = w x (1 - metric_CV) + (1 - w) x (số feature chọn / tổng feature)`, cần **tối thiểu hóa**. Mặc định `metric = f1`, `w = 0.99`.
3. **Đánh giá cuối**: huấn luyện lại với feature đã chọn trên phần train (`final_train_scope`), chấm trên test, cho `wrapper_model` và các classifier trong `extra_final_models` (vd `["lightgbm"]`).
4. **Baseline** dùng toàn bộ feature chạy 1 lần cho mỗi (dataset, seed), ngoài vòng lặp thuật toán, cùng train/test với các thuật toán của seed đó.
5. `seeds` (mặc định 10) quyết định cùng lúc cách chia dữ liệu, các fold CV và random của thuật toán.

**Phân chia trách nhiệm**: `pipeline.py` chỉ chạy và trả về dict kết quả thô (chất lượng, số feature, thời gian, đường hội tụ, mask). `main.py` điều phối vòng lặp, lưu `raw_results.pkl` rồi gọi `evaluate.py`. Toàn bộ việc đọc, in, so sánh, vẽ nằm trong `evaluate.py`, nên sửa cách báo cáo không cần chạy lại thí nghiệm.

## Tiền xử lý (`preprocess.py`)

Giai đoạn A, một lần cho mỗi dataset (không phụ thuộc seed), có các bước kiểm tra tường minh, sai thì báo lỗi ngay chứ không âm thầm bỏ:
1. Nhãn phải có đúng 2 giá trị và không thiếu; `positive_label` được mã hóa thành 1, giá trị còn lại thành 0.
2. Loại cột target và các cột trong `spec["drop_cols"]` (vd `url` của `hannousse`); số feature còn lại phải bằng `spec["n_features"]` (87 và 111).
3. Mọi cột feature còn lại phải là số. Cột không phải số làm báo lỗi kèm tên cột và ví dụ giá trị, không bị bỏ ngầm bằng `select_dtypes`.
4. Hàm làm sạch riêng của dataset (`CLEANERS`, hiện là khung `clean_passthrough`): chạy trước khi chia nên chỉ làm việc không học gì từ dữ liệu và không dùng nhãn (bỏ dòng trùng, bỏ cột, đổi giá trị đặc biệt như -1 hoặc số âm thành NaN, clip, log1p). Cột và dòng bị bỏ ở đây tự vào báo cáo.
5. Bỏ cột hằng số (`drop_constant_features`, mặc định bật).

Báo cáo của từng dataset (in khi bắt đầu chạy, có trong `report.txt`, lưu ở `preprocess_report.json` và `data/processed/<dataset>/preprocess_report.json`): số feature ban đầu, số feature bị loại, **danh sách feature bị loại kèm lý do**, số feature còn lại, số dòng bị loại, số ô NaN sẽ impute. Việc loại feature được làm ở giai đoạn A (giống nhau ở mọi seed) để có thể so sánh tập feature giữa các seed.


## Thuật toán (đều nhị phân, dùng chung Tracker nên được so sánh công bằng)

- **SA**: một nghiệm, mỗi bước lật `n_flip` bit; nhận nghiệm xấu hơn với xác suất exp(-delta/T); T giảm hình học (`t0`, `cooling`).
- **GA**: tournament, lai ghép đồng nhất, đột biến lật bit, giữ elite.
- **PSO**: PSO nhị phân (Kennedy & Eberhart): vận tốc thực, vị trí = 1 với xác suất sigmoid(v).
- **ACO**: mỗi feature có pheromone cho "chọn" / "bỏ"; kiến xây nghiệm theo xác suất từ pheromone; bay hơi `rho`, kiến tốt nhất vòng đó thêm pheromone, cắt trong `[tau_min, tau_max]` (kiểu MAX-MIN).
- **RS** (random search, đang tắt trong config): mốc ngẫu nhiên nếu muốn thêm.

Tham số nằm trong `CONFIG["optimizers"]`; chưa tinh chỉnh, cần chỉnh theo ngân sách (đặc biệt `t0` của SA, cỡ chênh lệch fitness giữa các nghiệm lân cận).

## Dừng, ngân sách và hội tụ

- **Ngân sách** `max_evals`: mặc định đếm theo **số lần gọi fitness** (`budget_on = "calls"`, kể cả cache hit), công bằng giữa các thuật toán. `"unique"` chỉ đếm lần huấn luyện thật.
- **Dừng sớm** `patience`: mặc định **tắt**. Tốc độ hội tụ vẫn được ghi (`evals_to_best`, `time_to_best_s`) và vẽ trong `convergence_<dataset>.png`.
- **Cache** theo mask, mỗi lần chạy một cache riêng; không đổi quỹ đạo hay nghiệm (đã kiểm tra với cả 4 thuật toán, cả SVM và RF), chỉ giảm thời gian. `time_s` là thời gian thật, `time_nocache_s` là ước tính nếu không cache. GA hưởng lợi nhiều nhất (cache hit 40-50%), PSO/ACO/SA gần như không.

## Kết quả và so sánh (không dùng kiểm định thống kê)

Báo cáo (`report.txt`, in ra màn hình) gồm, cho từng dataset, mean ± std qua các seed:

1. **Chất lượng nghiệm**: `best_fitness`, `search_score` (metric CV lúc tìm kiếm), metric trên test (`final_<model>_<metric>`), `gen_gap` = search_score - test.
2. **Giảm feature**: `n_selected`, `reduction_pct`.
3. **Chi phí và hội tụ**: `n_evals`, `n_fits`, `cache_hit_rate`, `evals_to_best`, `time_to_best_s`, `time_s`, `time_nocache_s`.
4. **Tổng quát hóa qua seed**: std của metric test, std số feature, độ ổn định tập feature (Jaccard trung bình các cặp seed; 1 = giống hệt).
5. **Tổng quát hóa qua dataset** (khi chạy 2 dataset): metric trung bình, metric của dataset tệ nhất, hạng trung bình (hạng 1 = tốt nhất trong từng dataset) cho chất lượng, giảm feature và thời gian.

Báo cáo còn có phần tiền xử lý từng dataset và dòng kiểm tra nhất quán dữ liệu (mọi kết quả cùng dataset và seed dùng cùng dấu vân tay dữ liệu).

File xuất ra: `report.txt`, `preprocess_report.json`, `summary.csv`, `stability.csv`, `cross_dataset.csv`, `runs.csv` (từng lần chạy), `selected_features.json`, `feature_frequency_<dataset>.csv`, `convergence_<dataset>.png`, `comparison_<dataset>.png`, `raw_results.pkl`.

Lưu ý khi diễn giải: Jaccard giữa các seed đo cả tính ngẫu nhiên của thuật toán lẫn độ nhạy với cách chia dữ liệu. Hạng trung bình qua 2 dataset chỉ có 2 mốc nên mang tính mô tả.

## Mở rộng

- **Thêm thuật toán** (vd GWO, WOA): trong `optimizers.py`, kế thừa `BaseOptimizer`, cài `search(self, evaluate, n_features, rng)`, thêm vào `OPTIMIZERS` và `CONFIG["optimizers"]`.
- **Thêm classifier**: trong `models.py`, viết hàm `make_xxx(params)`, thêm vào `MODELS` (và `WRAPPER_POOL` nếu cho dùng trong wrapper) và `CONFIG["models"]`. Cố định `random_state` (cần cho cache).
- **Thêm metric**: thêm hàm `f(y_true, y_pred)` vào `METRICS` trong `metrics.py`, rồi đặt `CONFIG["metric"]`.
- **Thêm bộ dữ liệu**: thêm mục vào `CONFIG["dataset_specs"]` (`raw_path`, `target_col`, `positive_label`, `drop_cols`, `n_features`), đăng ký hàm làm sạch (có thể dùng `clean_passthrough`) vào `CLEANERS`, thêm tên vào `CONFIG["datasets"]`.
- **Xử lý riêng cho dataset**: sửa hàm làm sạch trong `CLEANERS` (nhận `X_df`, `y`, `spec`; trả về `(X_df, y)`, toàn số). Impute và scale đã có sẵn ở giai đoạn B, không cần làm lại.

## Dùng trong notebook (demo)

```python
from src.main import run_scenario
results, out_dir = run_scenario("quick")
results, out_dir = run_scenario("compare", overrides={"seeds": [0, 1, 2], "wrapper_model": "rf",
                                                      "datasets": ["hannousse"]})
```
