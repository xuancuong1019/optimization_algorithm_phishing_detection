"""
models.py - Các classifier.

  * Dùng trong vòng wrapper (chọn 1 qua cfg["wrapper_model"]): các model trong WRAPPER_POOL (SVM, RF).
  * Dùng ở bước đánh giá cuối: wrapper_model và các model thêm trong cfg["extra_final_models"]
    (vd LightGBM, thường quá chậm để đưa vào vòng wrapper).

Thêm classifier mới: viết hàm nhận `params` (dict) và trả về estimator sklearn (có fit/predict),
đăng ký vào MODELS (và vào WRAPPER_POOL nếu cho phép dùng trong wrapper), rồi thêm tham số vào CONFIG["models"].
Tham số lấy từ cfg["models"][tên]. Nên cố định random_state để fitness xác định (cần cho cache).
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def make_svm(params):
    return SVC(**params)


def make_rf(params):
    return RandomForestClassifier(**params)         # cây không cần chuẩn hóa dữ liệu


def make_lightgbm(params):
    try:
        from lightgbm import LGBMClassifier         # import khi dùng, để không bắt buộc cài nếu không dùng
    except ImportError as e:
        raise ImportError("Cần cài LightGBM để dùng 'lightgbm': pip install lightgbm") from e
    return LGBMClassifier(**params)


MODELS = {
    "svm": make_svm,
    "rf": make_rf,
    "lightgbm": make_lightgbm,
}

# Các model được phép dùng trong vòng wrapper (cfg["wrapper_model"])
WRAPPER_POOL = ["svm", "rf"]


def get_model(name, params=None):
    return MODELS[name](params or {})
