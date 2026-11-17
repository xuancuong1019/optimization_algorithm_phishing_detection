"""
main.py - Điều phối: chạy toàn bộ thí nghiệm theo một kịch bản, rồi giao kết quả cho evaluate.py.

Chạy từ thư mục gốc của project:
    python -m src.main --scenario quick
    python -m src.main --scenario compare
    python -m src.main --scenario compare --wrapper rf
    python -m src.main --scenario compare --extra-final lightgbm
    python -m src.main --scenario quick --datasets hannousse

Chạy từ notebook (ví dụ để demo):
    from src.main import run_scenario
    results, out_dir = run_scenario("quick", overrides={"seeds": [0, 1, 2], "metric": "accuracy"})

Cấu trúc: mỗi dataset được tiền xử lý một lần (kiểm tra, làm sạch, báo cáo feature bị loại); với mỗi (dataset, seed)
preprocess chia train/test và impute/scale, rồi baseline và từng thuật toán chạy trên CÙNG một bộ dữ liệu đó qua
pipeline.py (chỉ trả kết quả). Cuối cùng evaluate.save_report tổng hợp, in và lưu mọi thứ vào
results/<kịch bản>_<thời gian>/.
"""
import argparse
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Cho phép chạy cả bằng `python src/main.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from threadpoolctl import threadpool_limits

from src.config import SCENARIOS, get_config
from src.evaluate import print_preprocess, save_report
from src.pipeline import (check_config, data_info, load_data, run_baseline, run_one,
                          split_data, warm_up)


def run_scenario(name="quick", overrides=None):
    """Chạy một kịch bản. Trả về (danh sách kết quả các lần chạy, thư mục kết quả)."""
    warnings.filterwarnings("ignore")                  # tắt cảnh báo hội tụ của sklearn cho gọn log
    cfg = get_config(name, overrides)
    check_config(cfg)

    # Tiền xử lý hết các dataset ngay từ đầu: thiếu file hoặc dữ liệu sai thì báo lỗi ngay, không phải chờ chạy xong dataset đầu
    bases = {d: load_data(d, cfg) for d in cfg["datasets"]}
    for base in bases.values():
        print_preprocess(base["report"])
    print(f"\nKịch bản '{name}': datasets {list(bases)}, {len(cfg['seeds'])} seed, "
          f"wrapper {cfg['wrapper_model']}, {len(cfg['optimizers'])} thuật toán, ngân sách {cfg['max_evals']}")

    results, baselines, meta = [], [], {}
    try:
        # Cố định số luồng BLAS cho cả thí nghiệm để thời gian đo ổn định (None = không giới hạn)
        with threadpool_limits(limits=cfg["blas_threads"]):
            for dataset, base in bases.items():
                for seed in cfg["seeds"]:
                    # Một bộ dữ liệu cho mỗi (dataset, seed): chia train/test, impute, scale, search set, fold CV.
                    # Baseline và mọi thuật toán bên dưới dùng chung đúng đối tượng này.
                    data = split_data(base, cfg, seed)
                    m = meta.setdefault(dataset, {"feature_names": base["feature_names"], "info": data_info(base, data),
                                                  "preprocess": base["report"], "splits": {}})
                    m["splits"][seed] = data["split_info"]
                    warm_up(cfg["wrapper_model"], data, cfg)

                    # Baseline (toàn bộ feature): 1 lần cho mỗi (dataset, seed), ngoài vòng lặp thuật toán
                    baselines.append(run_baseline(dataset, data, cfg, seed))

                    for opt_name in cfg["optimizers"]:
                        results.append(run_one(dataset, opt_name, data, cfg, seed))
                        print(f"[{dataset}] seed {seed} {opt_name}: xong ({results[-1]['time_s']:.0f}s)")
    except KeyboardInterrupt:
        print("\nĐã dừng giữa chừng: báo cáo chỉ gồm các lần chạy đã xong.")

    if not results:
        print("Chưa có kết quả nào để báo cáo.")
        return results, None

    out_dir = cfg["results_dir"] / f"{name}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Lưu kết quả thô để có thể chạy lại evaluate (python -m src.evaluate <thư mục>) mà không phải chạy lại thí nghiệm
    with open(out_dir / "raw_results.pkl", "wb") as f:
        pickle.dump({"results": results, "baselines": baselines, "cfg": cfg, "meta": meta}, f)
    save_report(results, baselines, cfg, meta, out_dir)
    return results, out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature selection bằng metaheuristic")
    parser.add_argument("--scenario", default="quick", choices=list(SCENARIOS), help="kịch bản cần chạy")
    parser.add_argument("--datasets", nargs="+", help="chỉ chạy các dataset này (mặc định: theo config)")
    parser.add_argument("--wrapper", choices=["svm", "rf"], help="classifier dùng trong vòng wrapper")
    parser.add_argument("--extra-final", nargs="+", help="classifier thêm ở bước đánh giá cuối, vd lightgbm")
    args = parser.parse_args()

    overrides = {}
    if args.datasets:
        overrides["datasets"] = args.datasets
    if args.wrapper:
        overrides["wrapper_model"] = args.wrapper
    if args.extra_final:
        overrides["extra_final_models"] = args.extra_final
    run_scenario(args.scenario, overrides)
