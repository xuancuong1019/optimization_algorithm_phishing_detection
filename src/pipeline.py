"""
pipeline.py - Luồng chạy cho MỘT (dataset, thuật toán, seed): chỉ chạy và TRẢ VỀ kết quả (không in, không so sánh).
Việc điều phối vòng lặp nằm ở main.py; việc đọc, in, so sánh kết quả nằm ở evaluate.py.

Các bước của một lần chạy (run_one):
  1. split_data      : nhận train/test từ preprocess.py, chọn search set, tạo các fold CV (theo seed); impute + scale
                       được fit lại cho TỪNG fold (chỉ trên phần train của fold) và cache một lần.
                       Mọi thuật toán cùng (dataset, seed) dùng chung đúng một đối tượng dữ liệu này (mảng chỉ đọc)
  2. build_fitness   : fitness(mask) = trung bình metric qua k fold (dữ liệu fold đã cache), cộng phạt số feature
  3. optimizer.run   : thuật toán tìm mask tốt nhất (có cache, đo thời gian)
  4. evaluate_subset : huấn luyện lần cuối trên phần train với feature đã chọn, chấm trên test (cho mỗi classifier cuối)
"""
import gc
import hashlib
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import final_models
from src.metrics import EMPTY_PENALTY, METRICS, fitness_value, get_metric, score_from_fitness
from src.models import MODELS, WRAPPER_POOL, get_model
from src.optimizers import OPTIMIZERS, get_optimizer
from src.preprocess import CLEANERS, fit_transform_pair, load_base, make_split


# ----------------------------------------------------------------------------
# Kiểm tra cấu hình (chạy một lần ở đầu, để lỗi cấu hình báo ngay thay vì sau nhiều giờ)
# ----------------------------------------------------------------------------
def check_config(cfg):
    if cfg["wrapper_model"] not in WRAPPER_POOL:
        raise ValueError(f"wrapper_model phải thuộc {WRAPPER_POOL}, hiện là '{cfg['wrapper_model']}'")
    for m in final_models(cfg):
        if m not in MODELS or m not in cfg["models"]:
            raise ValueError(f"Classifier '{m}' chưa được đăng ký trong models.MODELS hoặc thiếu trong cfg['models']")
        get_model(m, cfg["models"][m])      # tạo thử: thiếu thư viện (vd lightgbm) hoặc sai tham số thì báo ngay
    for o in cfg["optimizers"]:
        if o not in OPTIMIZERS:
            raise ValueError(f"Thuật toán '{o}' chưa được đăng ký trong optimizers.OPTIMIZERS: {list(OPTIMIZERS)}")
    for d in cfg["datasets"]:
        if d not in cfg["dataset_specs"] or d not in CLEANERS:
            raise ValueError(f"Dataset '{d}' cần có trong cfg['dataset_specs'] và preprocess.CLEANERS")
    if cfg["impute_strategy"] not in (None, "mean", "median", "most_frequent"):
        raise ValueError("impute_strategy phải là None, 'mean', 'median' hoặc 'most_frequent'")
    if cfg["metric"] not in METRICS:
        raise ValueError(f"metric phải thuộc {list(METRICS)}")
    if not 0 < cfg["w"] <= 1:
        raise ValueError("w phải nằm trong (0, 1]")
    if cfg["cv_folds"] < 2:
        raise ValueError("cv_folds phải >= 2")
    if cfg["final_train_scope"] not in ("full_train", "rest"):
        raise ValueError("final_train_scope phải là 'full_train' hoặc 'rest'")
    if cfg["budget_on"] not in ("calls", "unique"):
        raise ValueError("budget_on phải là 'calls' hoặc 'unique'")


# ----------------------------------------------------------------------------
# Dữ liệu
# ----------------------------------------------------------------------------
def load_data(name, cfg):
    """Giai đoạn A của preprocess (kiểm tra, làm sạch, báo cáo feature). Gọi 1 lần cho mỗi dataset."""
    return load_base(name, cfg)


def split_data(base, cfg, seed):
    """Lấy train/test từ preprocess.make_split, rồi chọn search set và tạo dữ liệu cho các fold CV (theo seed).

        preprocess : tổng = test (test_ratio) + train; impute/scale fit trên train (dùng cho bước đánh giá cuối)
        ở đây      : train = search (min(search_ratio * |train|, search_max_samples), dùng cho wrapper) + rest
                     final train = toàn bộ train ("full_train") hoặc chỉ rest ("rest")
                     CV của fitness: với MỖI fold, impute + scale được fit lại chỉ trên phần train của fold đó
                     (fold validation không ảnh hưởng tham số), rồi dữ liệu đã transform được CACHE trong
                     data["fold_data"] và dùng lại cho mọi lần đánh giá mask. Chi phí preprocessing là
                     cv_folds lần cho mỗi (dataset, seed), không phải số lần đánh giá fitness lần.
    """
    sp = make_split(base, cfg, seed)
    X, y = sp["X_train"], sp["y_train"]            # đã impute/scale bằng tham số của toàn bộ train: chỉ dùng cho bước cuối
    X_raw = sp["X_train_raw"]                      # chưa impute/scale: dùng để fit lại preprocessing cho từng fold
    train_idx = np.arange(len(y))

    n_search = int(round(cfg["search_ratio"] * len(train_idx)))
    if cfg["search_max_samples"]:
        n_search = min(n_search, cfg["search_max_samples"])
    if n_search >= len(train_idx):
        search_idx, rest_idx = train_idx, np.array([], dtype=int)
    else:
        search_idx, rest_idx = train_test_split(
            train_idx, train_size=n_search, stratify=y[train_idx], random_state=seed)

    if cfg["final_train_scope"] == "full_train":
        final_idx = train_idx
    else:  # "rest"
        if len(rest_idx) == 0:
            raise ValueError("final_train_scope='rest' cần search_ratio < 1")
        final_idx = rest_idx
    if cfg["final_max_samples"] and len(final_idx) > cfg["final_max_samples"]:
        final_idx, _ = train_test_split(final_idx, train_size=cfg["final_max_samples"],
                                        stratify=y[final_idx], random_state=seed)

    # Fold CV cho fitness: chia MỘT LẦN cho mỗi seed, mọi ứng viên dùng chung (tránh fitness bị nhiễu).
    # Mỗi fold: fit impute + scale chỉ trên phần train của fold, transform cả phần validation, rồi cache.
    X_search_raw, y_search = X_raw[search_idx], y[search_idx]
    skf = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)
    fold_data = []
    for tr, va in skf.split(X_search_raw, y_search):
        X_tr, X_va = fit_transform_pair(X_search_raw[tr], X_search_raw[va], cfg, base["name"])
        fold_data.append((X_tr, y_search[tr], X_va, y_search[va]))

    info = {**sp["info"], "fold_preprocess": f"impute/scale fit lại trên phần train của từng fold ({cfg['cv_folds']} lần)"}
    data = {
        "y_search": y_search, "fold_data": fold_data,    # không giữ X_search đã scale bằng toàn train, tránh dùng nhầm
        "X_final": X[final_idx], "y_final": y[final_idx],
        "X_test": sp["X_test"], "y_test": sp["y_test"],
        "n_features": X.shape[1], "feature_names": sp["feature_names"], "split_info": info,
    }
    # Dữ liệu dùng chung cho mọi thuật toán của (dataset, seed): đặt chỉ đọc để không thuật toán nào sửa được,
    # và lấy "dấu vân tay" để evaluate kiểm tra mọi kết quả cùng (dataset, seed) dùng đúng cùng dữ liệu.
    for a in _all_arrays(data):
        a.flags.writeable = False
    data["fingerprint"] = data_fingerprint(data)
    return data


def _all_arrays(data):
    """Mọi mảng thuật toán nhìn thấy: nhãn search, dữ liệu từng fold, train cuối, test."""
    arrays = [data[k] for k in ("y_search", "X_final", "y_final", "X_test", "y_test")]
    for fold in data["fold_data"]:
        arrays += list(fold)
    return arrays


def data_fingerprint(data):
    """Mã băm ngắn của toàn bộ dữ liệu thuật toán nhìn thấy: ma trận X (đã transform), nhãn, dữ liệu từng fold CV."""
    h = hashlib.sha1()
    for a in _all_arrays(data):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:12]


def data_info(base, data):
    """Kích thước dữ liệu và các phần chia (để in trong báo cáo)."""
    return {"n_samples": len(base["y"]), "n_features": data["n_features"],
            "n_search": len(data["y_search"]), "n_final_train": len(data["y_final"]),
            "n_test": len(data["y_test"]), "phishing_rate": float(base["y"].mean())}


# ----------------------------------------------------------------------------
# Fitness
# ----------------------------------------------------------------------------
def build_fitness(model_name, data, cfg):
    """Trả về hàm fitness(mask) -> số (cần tối thiểu hóa)."""
    metric = get_metric(cfg["metric"])
    params = cfg["models"][model_name]
    fold_data = data["fold_data"]               # đã impute/scale riêng cho từng fold (cache, tính một lần)
    n_features = data["n_features"]

    def fitness(mask):
        k = int(mask.sum())
        if k == 0:
            return EMPTY_PENALTY
        scores = []
        for X_tr, y_tr, X_va, y_va in fold_data:
            model = get_model(model_name, params)
            model.fit(X_tr[:, mask], y_tr)      # impute/scale theo từng cột độc lập: chọn cột sau khi transform là đủ
            scores.append(metric(y_va, model.predict(X_va[:, mask])))
        return fitness_value(float(np.mean(scores)), k, n_features, cfg["w"])

    return fitness


def warm_up(model_name, data, cfg):
    """Chạy thử một lần fitness trước khi đo thời gian, để lần chạy đầu không bị chậm vì khởi động."""
    build_fitness(model_name, data, cfg)(np.ones(data["n_features"], dtype=bool))


# ----------------------------------------------------------------------------
# Đánh giá cuối
# ----------------------------------------------------------------------------
def evaluate_subset(mask, data, cfg):
    """Huấn luyện từng classifier cuối trên phần train với feature được chọn, chấm trên test.

    Trả về {tên_model: {tên_metric: giá_trị}}.
    """
    out = {}
    for name in final_models(cfg):
        model = get_model(name, cfg["models"][name])
        model.fit(data["X_final"][:, mask], data["y_final"])
        pred = model.predict(data["X_test"][:, mask])
        out[name] = {m: fn(data["y_test"], pred) for m, fn in METRICS.items()}
    return out


# ----------------------------------------------------------------------------
# Một lần chạy thuật toán + baseline
# ----------------------------------------------------------------------------
def run_one(dataset, optimizer_name, data, cfg, seed):
    """Chạy một thuật toán một lần với một seed. Trả về dict kết quả (không in gì)."""
    model_name = cfg["wrapper_model"]
    fitness_fn = build_fitness(model_name, data, cfg)
    optimizer = get_optimizer(optimizer_name, cfg["optimizers"][optimizer_name])
    n_features = data["n_features"]

    gc.collect()                                       # tránh GC của lần trước chen vào phép đo
    t_wall, t_cpu = time.perf_counter(), time.process_time()
    out = optimizer.run(fitness_fn, n_features, seed, cfg["max_evals"], cfg["patience"],
                        cfg["min_delta"], cfg["budget_on"], cfg["use_cache"])
    time_s = time.perf_counter() - t_wall
    cpu_s = time.process_time() - t_cpu

    # Hội tụ: lần gọi đầu tiên đạt nghiệm tốt nhất cuối cùng (sai lệch trong min_delta)
    hist = np.array(out["history"])
    best_idx = int(np.argmax(hist <= hist[-1] + cfg["min_delta"]))
    n_calls, n_fits = out["n_calls"], out["n_fits"]

    mask = out["best_mask"]
    k = int(mask.sum())
    return {
        "dataset": dataset, "optimizer": optimizer_name, "seed": seed, "wrapper_model": model_name,
        "data_fingerprint": data["fingerprint"],       # mã băm dữ liệu đã dùng (evaluate kiểm tra tính nhất quán)
        # Chất lượng nghiệm trong lúc tìm kiếm
        "best_fitness": out["best_fitness"],
        "search_score": score_from_fitness(out["best_fitness"], k, n_features, cfg["w"]),  # metric CV trên search set
        # Số feature
        "n_selected": k, "n_features": n_features,
        # Chi phí và tốc độ hội tụ
        "n_evals": n_calls,                            # số lần gọi fitness (tính cả cache hit)
        "n_fits": n_fits,                              # số lần huấn luyện thật
        "cache_hit_rate": 1.0 - n_fits / n_calls,
        "stop_reason": out["stop_reason"],
        "evals_to_best": best_idx + 1,
        "time_to_best_s": out["times"][best_idx],
        "time_s": time_s,                              # thời gian thật của phần tìm kiếm (có cache nếu bật)
        "cpu_s": cpu_s,
        # Ước tính thời gian nếu không cache = thời gian thật - thời gian fitness đã tính thật
        #                                      + thời gian fitness nếu mọi lần gọi đều tính lại
        "time_nocache_s": time_s - out["fit_time_actual"] + out["fit_time_nocache"],
        # Chất lượng trên test (cho từng classifier cuối): {model: {metric: value}}
        "final": evaluate_subset(mask, data, cfg),
        # Dữ liệu thô cho biểu đồ hội tụ và so sánh tập feature
        "history": out["history"], "times": out["times"], "mask": mask,
    }


def run_baseline(dataset, data, cfg, seed):
    """Mốc so sánh: dùng TOÀN BỘ feature, không tìm kiếm. Chạy 1 lần cho mỗi (dataset, seed),
    nằm ngoài vòng lặp thuật toán. Dùng cùng train/test với các thuật toán của seed đó."""
    mask = np.ones(data["n_features"], dtype=bool)
    nan = np.nan
    return {
        "dataset": dataset, "optimizer": "all_features", "seed": seed, "wrapper_model": cfg["wrapper_model"],
        "data_fingerprint": data["fingerprint"],
        "best_fitness": nan, "search_score": nan,
        "n_selected": data["n_features"], "n_features": data["n_features"],
        "n_evals": nan, "n_fits": nan, "cache_hit_rate": nan, "stop_reason": "-",
        "evals_to_best": nan, "time_to_best_s": nan, "time_s": nan, "cpu_s": nan, "time_nocache_s": nan,
        "final": evaluate_subset(mask, data, cfg),
        "history": [], "times": [], "mask": mask,
    }