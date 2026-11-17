"""
metrics.py - Các metric (để chọn cho hàm fitness và để báo cáo) và công thức fitness.

Mọi metric có dạng f(y_true, y_pred) -> số, LỚN HƠN = TỐT HƠN.
Thêm metric mới: viết hàm cùng dạng rồi đăng ký vào METRICS.
"""
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score)

METRICS = {
    "accuracy": accuracy_score,
    "precision": lambda y, p: precision_score(y, p, zero_division=0),
    "recall": lambda y, p: recall_score(y, p, zero_division=0),
    "f1": lambda y, p: f1_score(y, p, zero_division=0),
    "balanced_accuracy": balanced_accuracy_score,
}

# Fitness của mask rỗng (không chọn feature nào): tệ hơn mọi giá trị hợp lệ (nằm trong [0, 1]).
EMPTY_PENALTY = 2.0


def get_metric(name):
    return METRICS[name]


def fitness_value(score, n_selected, n_features, w):
    """Fitness cần TỐI THIỂU HÓA.

    fitness = w * (1 - score) + (1 - w) * (n_selected / n_features)
      - w gần 1: ưu tiên chất lượng phân loại
      - w nhỏ hơn: ưu tiên chọn ít feature hơn
    """
    return w * (1.0 - score) + (1.0 - w) * (n_selected / n_features)


def score_from_fitness(fitness, n_selected, n_features, w):
    """Suy ngược metric (trung bình CV trên search set) từ fitness của một mask: nghịch đảo của fitness_value."""
    return 1.0 - (fitness - (1.0 - w) * n_selected / n_features) / w
