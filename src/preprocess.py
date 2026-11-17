"""
preprocess.py - Tiền xử lý: kiểm tra đầu vào, làm sạch riêng từng dataset, chia train/test, impute và scale.

Hai giai đoạn:

  A. load_base(name, cfg)  - MỘT lần cho mỗi dataset, không phụ thuộc seed
       1. đọc CSV; kiểm tra nhãn đúng 2 giá trị và mã hóa thành 0/1 (positive_label -> 1)
       2. loại cột target và các cột trong spec["drop_cols"] (vd 'url'); kiểm tra số feature còn lại == spec["n_features"]
       3. kiểm tra các cột còn lại phải là số: cột không phải số làm BÁO LỖI, không bị âm thầm bỏ
       4. hàm làm sạch riêng của dataset (CLEANERS) - NƠI THÀNH VIÊN PHỤ TRÁCH DỮ LIỆU BỔ SUNG XỬ LÝ
       5. bỏ cột hằng số (tùy chọn drop_constant_features)
       -> trả về base gồm bảng feature sạch, nhãn và báo cáo (số feature ban đầu, số bị loại, danh sách bị loại)

  B. make_split(base, cfg, seed) - mỗi seed một lần
       1. chia train/test (stratified, theo seed)
       2. impute và scale: FIT TRÊN TRAIN, sau đó chỉ TRANSFORM test bằng tham số của train (không fit trên test)
       -> trả về X_train, y_train, X_test, y_test đã xử lý (dùng cho bước đánh giá cuối),
          kèm X_train_raw (CHƯA impute/scale) để pipeline fit lại preprocessing cho từng fold CV

  C. fit_transform_pair(X_fit, X_other, cfg) - cùng một định nghĩa impute + scale cho mọi nơi dùng
       Fit CHỈ trên X_fit rồi áp cho cả X_fit và X_other. make_split dùng cho (train, test); pipeline.split_data dùng
       cho (phần train của fold, phần validation của fold) để fold validation không ảnh hưởng tham số impute/scale
       (tránh rò rỉ trong CV). Tính một lần cho mỗi fold rồi cache, không tính lại ở mỗi lần đánh giá fitness.
       Impute và scale đều theo từng cột độc lập, nên chọn cột (mask) SAU khi transform cho kết quả y hệt
       việc transform riêng các cột được chọn.

Mọi thuật toán trong cùng (dataset, seed) nhận đúng cùng ma trận này (pipeline.py dùng chung một đối tượng dữ liệu).

Hàm làm sạch riêng (CLEANERS) chạy TRƯỚC khi chia nên chỉ được làm các việc không học gì từ dữ liệu và không dùng nhãn
để quyết định: bỏ dòng trùng, bỏ cột, đổi giá trị đặc biệt (vd -1, số âm) thành NaN, biến đổi từng giá trị (log1p, clip).
Những gì cần học từ dữ liệu (giá trị impute, scaler) KHÔNG đặt ở đây vì đã có bước B fit trên train.
Việc loại feature phải giống nhau ở mọi seed (để so sánh tập feature giữa các seed), nên không loại feature ở bước B.

Thêm bộ dữ liệu mới: thêm mục vào CONFIG["dataset_specs"], (nếu cần) viết hàm làm sạch và đăng ký vào CLEANERS.
"""
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------
# Hàm làm sạch riêng cho từng dataset (khung)
# ----------------------------------------------------------------------------
def clean_passthrough(X_df, y, spec):
    """Chưa xử lý gì riêng. Nhận bảng feature (toàn số) và nhãn 0/1, trả về (X_df, y) sau khi làm sạch.

    
    Cột bị bỏ và dòng bị bỏ ở đây sẽ tự được ghi vào báo cáo.
    """
    return X_df, y


# Đăng ký: tên dataset -> hàm làm sạch riêng
CLEANERS = {
    "hannousse": clean_passthrough,   # TODO: bổ sung xử lý riêng khi nhóm chốt
    "vrbancic": clean_passthrough,    # TODO: bổ sung xử lý riêng khi nhóm chốt (vd giá trị -1 = thành phần URL không tồn tại)
}


# ----------------------------------------------------------------------------
# Giai đoạn A: kiểm tra và làm sạch (không phụ thuộc seed)
# ----------------------------------------------------------------------------
def encode_label(series, spec):
    """Nhãn phải có đúng 2 giá trị (không thiếu); positive_label -> 1, giá trị còn lại -> 0."""
    target = spec["target_col"]
    if series.isna().any():
        raise ValueError(f"Cột nhãn '{target}' có {int(series.isna().sum())} giá trị thiếu")
    values = series.unique().tolist()
    if len(values) != 2:
        raise ValueError(f"Cột nhãn '{target}' phải có đúng 2 giá trị, hiện có {len(values)}: {values[:10]}")
    if spec["positive_label"] not in values:
        raise ValueError(f"positive_label={spec['positive_label']!r} không nằm trong các giá trị nhãn {values}")
    return (series == spec["positive_label"]).astype(int).to_numpy()


def check_numeric(X_df, context):
    """Mọi cột feature phải là số. Cột không phải số làm báo lỗi (kèm ví dụ giá trị) thay vì bị âm thầm bỏ."""
    bad = [c for c in X_df.columns if not (pd.api.types.is_numeric_dtype(X_df[c]) or pd.api.types.is_bool_dtype(X_df[c]))]
    if bad:
        detail = "; ".join(f"'{c}' (vd {X_df[c].dropna().unique()[:3].tolist()})" for c in bad[:10])
        raise ValueError(
            f"{context}: {len(bad)} cột feature không phải số: {detail}.\n"
            f"Cột định danh (URL, id...) -> thêm vào spec['drop_cols']. Cột số bị đọc thành chuỗi "
            f"(vd có ký tự lạ) -> sửa trong hàm làm sạch của dataset.")


def load_base(name, cfg):
    """Giai đoạn A. Trả về base = {'name', 'X_df', 'y', 'feature_names', 'report'}."""
    spec = cfg["dataset_specs"][name]
    path = spec["raw_path"]
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu gốc của '{name}': {path}\n"
            f"Hãy đặt file CSV vào đúng đường dẫn trên (xem README), hoặc chạy bộ còn lại bằng "
            f"--datasets <tên> / overrides={{'datasets': [...]}}.")
    df = pd.read_csv(path)
    n_raw_rows, n_raw_cols = df.shape

    # 1. Nhãn
    target = spec["target_col"]
    if target not in df.columns:
        raise KeyError(f"Dataset '{name}': không có cột nhãn '{target}'. Các cột: {list(df.columns)[:10]}...")
    y = encode_label(df[target], spec)

    # 2. Loại target và các cột không phải feature (vd 'url'); kiểm tra số feature
    drop_cols = list(spec.get("drop_cols", []))
    missing = [c for c in drop_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Dataset '{name}': spec['drop_cols'] có cột không tồn tại trong file: {missing}")
    X_df = df.drop(columns=[target] + drop_cols)
    n_initial = X_df.shape[1]
    expected = spec.get("n_features")
    if expected is not None and n_initial != expected:
        raise ValueError(f"Dataset '{name}': sau khi loại target và {drop_cols} còn {n_initial} cột feature, "
                         f"kỳ vọng {expected}. Kiểm tra file dữ liệu hoặc spec['n_features'] / spec['drop_cols'].")

    # 3. Cột không phải số -> báo lỗi
    check_numeric(X_df, f"Dataset '{name}'")
    X_df = X_df.astype(float).replace([np.inf, -np.inf], np.nan)
    n_inf = int(np.isinf(df.drop(columns=[target] + drop_cols).select_dtypes(include="number")).sum().sum())

    # 4. Làm sạch riêng của dataset
    initial_cols = list(X_df.columns)
    X_df, y = CLEANERS[name](X_df.copy(), y.copy(), spec)
    X_df, y = X_df.reset_index(drop=True), np.asarray(y).astype(int)
    if len(X_df) != len(y):
        raise ValueError(f"Hàm làm sạch của '{name}' trả về số dòng feature ({len(X_df)}) khác số nhãn ({len(y)})")
    check_numeric(X_df, f"Dataset '{name}' sau hàm làm sạch")
    X_df = X_df.astype(float).replace([np.inf, -np.inf], np.nan)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Dataset '{name}': sau làm sạch nhãn phải còn đủ cả 2 lớp 0 và 1")
    dropped = [{"name": c, "reason": "bị loại bởi hàm làm sạch riêng của dataset"}
               for c in initial_cols if c not in X_df.columns]

    # 5. Cột hằng số (không dùng nhãn; hằng số trên toàn dữ liệu thì hằng số trên mọi cách chia)
    if cfg["drop_constant_features"]:
        const = [c for c in X_df.columns if X_df[c].nunique(dropna=True) <= 1]
        dropped += [{"name": c, "reason": "hằng số (chỉ 1 giá trị duy nhất)"} for c in const]
        X_df = X_df.drop(columns=const)
    if X_df.shape[1] == 0:
        raise ValueError(f"Dataset '{name}': không còn feature nào sau tiền xử lý")

    report = {
        "dataset": name, "raw_path": str(path),
        "n_raw_rows": n_raw_rows, "n_raw_columns": n_raw_cols,
        "non_feature_columns": [target] + drop_cols,
        "n_features_initial": n_initial,                      # sau khi loại target và drop_cols
        "n_features_dropped": len(dropped),
        "dropped_features": dropped,                          # [{name, reason}]
        "n_features_final": X_df.shape[1],                    # số feature đưa vào thuật toán tối ưu
        "n_rows_final": len(y), "n_rows_dropped": n_raw_rows - len(y),
        "n_inf_cells_to_nan": n_inf,
        "n_nan_cells": int(X_df.isna().sum().sum()),          # sẽ được impute ở make_split
        "label_encoding": {"1": spec["positive_label"],
                           "0": [v for v in df[target].unique().tolist() if v != spec["positive_label"]][0]},
        "n_positive": int(y.sum()), "n_negative": int((1 - y).sum()),
    }
    out_dir = cfg["processed_dir"] / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preprocess_report.json").write_text(
        json.dumps({**report, "feature_names": list(X_df.columns)}, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    return {"name": name, "X_df": X_df, "y": y, "feature_names": list(X_df.columns), "report": report}


# ----------------------------------------------------------------------------
# Giai đoạn B: chia train/test, impute, scale (theo seed)
# ----------------------------------------------------------------------------
def fit_transform_pair(X_fit, X_other, cfg, name=""):
    """Impute + scale với tham số học CHỈ từ X_fit, áp cho cả X_fit và X_other. Trả về (X_fit', X_other').

    Dùng cho (train, test) trong make_split và cho (train của fold, validation của fold) trong pipeline.split_data.
    """
    steps = []
    if cfg["impute_strategy"]:
        steps.append(("imputer", SimpleImputer(strategy=cfg["impute_strategy"], keep_empty_features=True)))
    if cfg["scale"]:
        steps.append(("scaler", StandardScaler()))
    if not steps:
        n_nan = int(np.isnan(X_fit).sum() + np.isnan(X_other).sum())
        if n_nan:
            raise ValueError(f"Dữ liệu '{name}' còn {n_nan} ô NaN nhưng impute_strategy=None")
        return X_fit, X_other
    transformer = Pipeline(steps).fit(X_fit)
    return transformer.transform(X_fit), transformer.transform(X_other)


def make_split(base, cfg, seed):
    """Chia train/test theo seed rồi impute + scale với tham số học từ TRAIN (test chỉ được transform).

    Trả về dict: X_train, y_train, X_test, y_test (đã xử lý), X_train_raw (chưa impute/scale), feature_names, info.
    """
    X, y = base["X_df"].to_numpy(dtype=float), base["y"]
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=cfg["test_ratio"],
                                           stratify=y, random_state=seed)
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    n_nan_train, n_nan_test = int(np.isnan(X_train_raw).sum()), int(np.isnan(X_test_raw).sum())
    X_train, X_test = fit_transform_pair(X_train_raw, X_test_raw, cfg, base["name"])

    out = {
        "X_train": X_train, "y_train": y[train_idx], "X_test": X_test, "y_test": y[test_idx],
        "X_train_raw": X_train_raw,                    # để pipeline fit lại preprocessing cho từng fold CV
        "feature_names": base["feature_names"],
        "info": {"seed": seed, "n_train": len(train_idx), "n_test": len(test_idx),
                 "nan_imputed_train": n_nan_train, "nan_imputed_test": n_nan_test,
                 "positive_rate_train": float(y[train_idx].mean()), "positive_rate_test": float(y[test_idx].mean()),
                 "impute": cfg["impute_strategy"], "scale": bool(cfg["scale"]),
                 "fit_on": "train (test chỉ được transform)"},
    }
    if seed in cfg["save_processed_seeds"]:
        d = cfg["processed_dir"] / base["name"]
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"seed{seed}.npz", X_train=out["X_train"], y_train=out["y_train"],
                            X_test=out["X_test"], y_test=out["y_test"],
                            feature_names=np.array(base["feature_names"]))
    return out