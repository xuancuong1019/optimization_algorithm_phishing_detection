"""
config.py - Toàn bộ cấu hình của project nằm ở đây.

- CONFIG    : cấu hình mặc định.
- SCENARIOS : mỗi kịch bản chỉ ghi các mục muốn GHI ĐÈ lên CONFIG.
- get_config(): trộn CONFIG + kịch bản + overrides (dùng khi chạy từ notebook).

Lưu ý: khi ghi đè một mục dạng dict (vd "optimizers", "models") thì dict mới
THAY THẾ hoàn toàn dict cũ, không trộn từng khóa con.
"""
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    # ---------- Dữ liệu ----------
    # Danh sách bộ dữ liệu có sẵn (tên -> thông tin file). Thêm bộ mới: thêm một mục ở đây
    # và đăng ký hàm tiền xử lý cùng tên trong preprocess.py.
    "dataset_specs": {
        # Hannousse & Yahiouche (2021): 11.430 mẫu, 87 đặc trưng, nhãn chữ 'phishing'/'legitimate'
        "hannousse": {
            "raw_path": ROOT / "data" / "raw" / "dataset_phishing.csv",
            "target_col": "status",
            "positive_label": "phishing",
            "drop_cols": ["url"],               # cột không phải feature (chuỗi định danh), loại tường minh
            "n_features": 87,                   # số feature kỳ vọng sau khi loại target và drop_cols (sai thì báo lỗi)
        },
        # Vrbancic et al. (2020), Mendeley Data 10.17632/72ptz43s9v.1: 111 đặc trưng, nhãn 0/1 'phishing'.
        # dataset_small (58.645 mẫu, cân bằng hơn) là mặc định; đổi sang dataset_full.csv (88.647 mẫu) nếu cần.
        "vrbancic": {
            "raw_path": ROOT / "data" / "raw" / "dataset_small.csv",
            "target_col": "phishing",
            "positive_label": 1,
            "drop_cols": [],
            "n_features": 111,
        },
    },
    "datasets": ["hannousse", "vrbancic"],      # các bộ dữ liệu sẽ chạy (tên trong dataset_specs)
    "processed_dir": ROOT / "data" / "processed",
    "results_dir": ROOT / "results",

    # ---------- Seed ----------
    # Mỗi seed quyết định CÙNG LÚC: cách chia train/test, cách chia các fold CV trong fitness,
    # và random của thuật toán. Mỗi (dataset, thuật toán) được chạy 1 lần cho mỗi seed.
    # Các thuật toán chạy cùng seed thì dùng đúng cùng dữ liệu.
    "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],

    # ---------- Tiền xử lý và chia train/test (preprocess.py; theo từng dataset và seed) ----------
    "test_ratio": 0.2,                          # 20% tổng dữ liệu là test, KHÔNG đụng tới khi tìm kiếm
    "impute_strategy": "median",                # impute NaN, fit trên train rồi áp cho test: mean | median | most_frequent | None
    "scale": True,                              # StandardScaler, fit trên train rồi áp cho test
    "drop_constant_features": True,             # bỏ cột hằng số (không dùng nhãn); danh sách bị loại được in và lưu
    "save_processed_seeds": [0],                # seed nào được lưu train/test đã xử lý ra data/processed/<dataset>/seed<k>.npz

    # ---------- Search set và fold CV (pipeline.py, trên phần train do preprocess trả về) ----------
    "search_ratio": 0.5,                        # wrapper chỉ dùng 50% của phần train để tìm kiếm
    "search_max_samples": 3000,                 # ...và tối đa ngần này mẫu (giữ chi phí fitness chấp nhận được
                                                #    trên dataset lớn; None = không giới hạn)
    # Huấn luyện lần cuối (cho thuật toán và cho baseline) trên:
    #   "full_train": toàn bộ phần train (80%)           <- mặc định
    #   "rest"     : chỉ phần train KHÔNG nằm trong search set
    "final_train_scope": "full_train",
    "final_max_samples": None,                  # giới hạn số mẫu khi huấn luyện lần cuối (None = không giới hạn).
                                                # SVM trên ~47.000 mẫu mất vài phút mỗi lần fit -> với dataset lớn
                                                # nên đặt (vd 15000); áp dụng như nhau cho baseline và mọi thuật toán
    "cv_folds": 3,                             # fitness = trung bình metric qua k fold (stratified) trên search set

    # ---------- Classifier (tên -> tham số; xem models.py) ----------
    # random_state cố định để fitness xác định (cần cho cache). n_jobs=1 để thời gian đo ổn định.
    "models": {
        "svm": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
        "rf": {"n_estimators": 50, "n_jobs": 1, "random_state": 0},
        "lightgbm": {"n_estimators": 200, "learning_rate": 0.1, "n_jobs": 1, "random_state": 0, "verbose": -1},
    },
    "wrapper_model": "svm",                     # classifier dùng trong vòng wrapper (chọn 1 trong "svm", "rf")
    "extra_final_models": [],                   # classifier thêm vào ở bước đánh giá cuối, vd ["lightgbm"]
                                                # (wrapper_model luôn được đánh giá cuối)

    # ---------- Fitness (xem metrics.py) ----------
    "metric": "f1",                             # accuracy | precision | recall | f1 | balanced_accuracy
    "w": 0.99,                                  # fitness = w*(1-metric) + (1-w)*(số feature chọn / tổng feature)

    # ---------- Thuật toán tối ưu (tên -> tham số; xem optimizers.py) ----------
    "optimizers": {
        "SA": {"t0": 0.01, "cooling": 0.995, "n_flip": 2, "init_prob": 0.5},
        "GA": {"pop_size": 30, "crossover_rate": 0.9, "mutation_rate": None,   # None -> 1 / số feature
               "tournament_size": 3, "elite_size": 2, "init_prob": 0.5},
        "PSO": {"n_particles": 30, "w": 0.7, "c1": 1.5, "c2": 1.5, "v_max": 4.0, "init_prob": 0.5},
        "ACO": {"n_ants": 30, "rho": 0.1, "tau_min": 0.05, "tau_max": 5.0},
        # "RS": {"p_select": 0.5},              # random search, bỏ comment nếu muốn thêm mốc so sánh ngẫu nhiên
    },

    # ---------- Tiêu chí dừng (dùng chung cho mọi thuật toán) ----------
    "max_evals": 2000,                          # ngân sách tối đa (cách đếm xem budget_on)
    # "calls"  : đếm số lần thuật toán GỌI hàm fitness (kể cả lần trúng cache)  <- mặc định, công bằng nhất
    # "unique" : chỉ đếm số lần huấn luyện thật (không tính cache hit); thuật toán hội tụ sẽ chạy nhiều vòng hơn
    "budget_on": "calls",
    "patience": None,                           # dừng sớm nếu N lần gọi liên tiếp không cải thiện; None = tắt (mặc định)
    "min_delta": 1e-4,                          # mức cải thiện tối thiểu được tính là "có cải thiện"
                                                # (cũng dùng để xác định 'evals_to_best' khi báo cáo hội tụ)

    # ---------- Cache & đo thời gian ----------
    "use_cache": True,                          # cache fitness theo mask (kết quả không đổi, chỉ nhanh hơn)
    "blas_threads": 1,                          # cố định số luồng BLAS để thời gian ổn định; None = không giới hạn
}

# Mỗi kịch bản chỉ liệt kê những gì khác với CONFIG.
SCENARIOS = {
    # Chạy thử nhanh để kiểm tra khung hoạt động (vài phút)
    "quick": {
        "seeds": [0, 1],
        "max_evals": 40,
        "search_max_samples": 1500,
        "optimizers": {
            "SA": {"t0": 0.01, "cooling": 0.995, "n_flip": 2},
            "GA": {"pop_size": 8, "elite_size": 1},
            "PSO": {"n_particles": 8},
            "ACO": {"n_ants": 8},
        },
    },
    # So sánh đầy đủ: 10 seed, dùng hết ngân sách, không dừng sớm (dùng đúng CONFIG)
    "compare": {},
    # Wrapper dùng Random Forest thay cho SVM
    "compare_rf": {"wrapper_model": "rf"},
    # Đánh giá cuối thêm LightGBM (cần: pip install lightgbm)
    "compare_with_lightgbm": {"extra_final_models": ["lightgbm"]},
    # Bật dừng sớm: xem thuật toán nào hội tụ sớm hơn mà không mất chất lượng
    "compare_early_stop": {"patience": 300},
    # Ngân sách đếm theo số lần huấn luyện thật (không tính cache hit)
    "compare_unique_budget": {"budget_on": "unique"},
    # Tắt cache: đo thời gian THẬT khi không cache (để đối chiếu với 'time_nocache_s' ước tính)
    "compare_no_cache": {"use_cache": False},
}


def get_config(scenario="compare", overrides=None):
    """Trả về cấu hình = CONFIG, rồi ghi đè bằng kịch bản, rồi bằng overrides."""
    if scenario not in SCENARIOS:
        raise KeyError(f"Kịch bản '{scenario}' không tồn tại. Có: {list(SCENARIOS)}")
    cfg = copy.deepcopy(CONFIG)
    cfg.update(copy.deepcopy(SCENARIOS[scenario]))
    cfg.update(copy.deepcopy(overrides or {}))
    return cfg


def final_models(cfg):
    """Danh sách classifier được đánh giá ở bước cuối: wrapper_model đứng đầu, rồi các model thêm."""
    return [cfg["wrapper_model"]] + [m for m in cfg["extra_final_models"] if m != cfg["wrapper_model"]]
