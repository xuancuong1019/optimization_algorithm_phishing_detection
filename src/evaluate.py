"""
evaluate.py - Đọc kết quả thô, tổng hợp, in và so sánh các thuật toán tối ưu, xuất báo cáo.

Đầu vào: danh sách kết quả của pipeline.run_one (results) và pipeline.run_baseline (baselines).
Mỗi kết quả là một dict gồm: chất lượng nghiệm, số feature, chi phí/tốc độ hội tụ, chất lượng trên test
(cho từng classifier cuối), cùng lịch sử hội tụ và mask feature được chọn.

Các khía cạnh được so sánh:
  1. Chất lượng nghiệm : fitness và metric CV trong lúc tìm kiếm, metric trên test (F1, accuracy...)
  2. Giảm feature      : số feature được chọn, % feature giảm được so với dùng toàn bộ
  3. Chi phí, hội tụ   : số lần gọi/huấn luyện, cache hit, thời gian, thời gian không cache (ước tính),
                         số lần gọi và thời gian để đạt nghiệm cuối, đường hội tụ
  4. Tổng quát hóa     : (a) độ lệch chuẩn qua các seed, (b) độ ổn định tập feature giữa các seed (Jaccard),
                         (c) khoảng chênh giữa metric lúc tìm kiếm và trên test, (d) thứ hạng qua các dataset

Ngoài ra: báo cáo tiền xử lý (số feature ban đầu, bị loại, danh sách bị loại) và kiểm tra mọi thuật toán cùng
(dataset, seed) dùng đúng cùng dữ liệu (so mã băm dữ liệu).

Đầu ra (trong out_dir): runs.csv, summary.csv, stability.csv, cross_dataset.csv, report.txt, preprocess_report.json,
  selected_features.json, feature_frequency_<dataset>.csv, convergence_<dataset>.png, comparison_<dataset>.png

Chạy lại báo cáo từ kết quả đã lưu (không cần chạy lại thí nghiệm):
    python -m src.evaluate results/<thư mục kết quả>
"""
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import final_models

BASELINE = "all_features"       # tên dòng baseline (dùng toàn bộ feature)
COST_COLS = ["n_evals", "n_fits", "cache_hit_rate", "evals_to_best",
             "time_to_best_s", "time_s", "time_nocache_s"]


def final_col(model, metric):
    """Tên cột metric trên test của một classifier, vd 'final_svm_f1'."""
    return f"final_{model}_{metric}"


# ----------------------------------------------------------------------------
# Chuẩn bị bảng dữ liệu
# ----------------------------------------------------------------------------
def results_to_df(results, baselines, cfg):
    """Gộp kết quả thành DataFrame phẳng (bỏ history/times/mask) và thêm các cột dẫn xuất."""
    rows = []
    for r in list(results) + list(baselines):
        row = {k: v for k, v in r.items() if k not in ("history", "times", "mask", "final")}
        for model, scores in r["final"].items():
            for metric, value in scores.items():
                row[final_col(model, metric)] = value
        rows.append(row)
    df = pd.DataFrame(rows)

    primary = final_col(cfg["wrapper_model"], cfg["metric"])
    df["reduction_pct"] = 100.0 * (1.0 - df["n_selected"] / df["n_features"])   # % feature giảm được
    df["gen_gap"] = df["search_score"] - df[primary]       # > 0: metric lúc tìm kiếm lạc quan hơn trên test

    df["dataset"] = pd.Categorical(df["dataset"], categories=cfg["datasets"], ordered=True)
    df["optimizer"] = pd.Categorical(df["optimizer"], categories=list(cfg["optimizers"]) + [BASELINE], ordered=True)
    return df


def column_groups(cfg):
    """Các nhóm cột đưa vào bảng báo cáo."""
    wm, metric = cfg["wrapper_model"], cfg["metric"]
    main = [final_col(wm, metric)]
    if metric != "accuracy":
        main.append(final_col(wm, "accuracy"))
    extra = [final_col(m, metric) for m in final_models(cfg)[1:]]     # classifier thêm (vd lightgbm)
    return {
        "1. CHẤT LƯỢNG NGHIỆM (search_score = metric CV lúc tìm kiếm; final_* = metric trên test; "
        "gen_gap = search_score - test)":
            ["best_fitness", "search_score"] + main + extra + ["gen_gap"],
        "2. GIẢM FEATURE": ["n_selected", "reduction_pct"],
        "3. CHI PHÍ VÀ TỐC ĐỘ HỘI TỤ (evals_to_best, time_to_best_s = lúc đạt nghiệm cuối; "
        "time_nocache_s = ước tính nếu không cache)": COST_COLS,
    }


def summarize(df, columns):
    """Bảng theo (dataset, thuật toán): <cột>_mean và <cột>_std qua các seed."""
    g = df.groupby(["dataset", "optimizer"], observed=True)[columns].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g


def feature_stability(results):
    """Độ ổn định tập feature giữa các seed: Jaccard trung bình của mọi cặp mask (1 = giống hệt nhau).

    Lưu ý: mỗi seed là một cách chia dữ liệu khác nhau, nên chỉ số này đo cả độ nhạy với dữ liệu lẫn tính ngẫu nhiên.
    """
    groups = {}
    for r in results:
        groups.setdefault((r["dataset"], r["optimizer"]), []).append(r["mask"])
    rows = []
    for (dataset, optimizer), masks in groups.items():
        sims = [(a & b).sum() / max((a | b).sum(), 1) for a, b in itertools.combinations(masks, 2)]
        rows.append({"dataset": dataset, "optimizer": optimizer,
                     "jaccard_mean": float(np.mean(sims)) if sims else np.nan, "n_runs": len(masks)})
    return pd.DataFrame(rows).set_index(["dataset", "optimizer"])


def stability_table(summary, stab, cfg):
    """Gộp các chỉ số ổn định: độ lệch chuẩn qua seed của metric test và của số feature, cùng Jaccard."""
    primary = final_col(cfg["wrapper_model"], cfg["metric"])
    s = summary[[f"{primary}_std", "n_selected_std"]].rename(
        columns={f"{primary}_std": f"{primary}_std_qua_seed", "n_selected_std": "n_selected_std_qua_seed"})
    return s.join(stab[["jaccard_mean"]], how="inner")


def cross_dataset_table(summary, cfg):
    """So sánh qua các dataset: giá trị trung bình và hạng trung bình (hạng 1 = tốt nhất) của từng thuật toán.

    Thuật toán tốt và tổng quát hóa tốt thì có hạng ổn định (thấp) trên mọi dataset, chứ không chỉ trên một dataset.
    """
    primary = final_col(cfg["wrapper_model"], cfg["metric"])
    d = summary.reset_index()
    d = d[d["optimizer"] != BASELINE]
    out = {}
    for col, higher_is_better, label in [(primary, True, "test"), ("reduction_pct", True, "reduction"),
                                         ("time_nocache_s", False, "time")]:
        mean = d.pivot_table(index="optimizer", columns="dataset", values=f"{col}_mean", observed=True)
        rank = mean.rank(ascending=not higher_is_better, method="min")
        out[f"{label}: {col} (TB qua dataset)"] = mean.mean(axis=1)
        out[f"{label}: hạng TB"] = rank.mean(axis=1)
        if label == "test":
            out[f"{label}: dataset tệ nhất"] = mean.min(axis=1)
            for ds in mean.columns:
                out[f"{label}: {ds}"] = mean[ds]
    return pd.DataFrame(out)


def check_data_consistency(results, baselines):
    """Mọi kết quả (thuật toán và baseline) cùng (dataset, seed) phải dùng đúng cùng dữ liệu: cùng ma trận X, cùng cách
    chia train/test/search/fold, cùng impute và scale. Kiểm tra qua mã băm dữ liệu (data_fingerprint) mà pipeline ghi lại.

    Trả về (ok, dòng_chữ).
    """
    groups = {}
    for r in list(results) + list(baselines):
        groups.setdefault((r["dataset"], r["seed"]), set()).add(r.get("data_fingerprint"))
    bad = [k for k, v in groups.items() if len(v) != 1 or None in v]
    if bad:
        return False, f"LỖI nhất quán dữ liệu: các (dataset, seed) {bad} có kết quả dùng dữ liệu khác nhau hoặc thiếu dấu vân tay"
    return True, (f"Nhất quán dữ liệu: OK ({len(groups)} cặp (dataset, seed); trong mỗi cặp, baseline và mọi thuật toán "
                  f"dùng cùng X, cùng cách chia và cùng impute/scale)")


def format_preprocess(rep):
    """Báo cáo tiền xử lý của một dataset: số feature ban đầu, số bị loại, danh sách bị loại."""
    lines = [
        f"TIỀN XỬ LÝ {rep['dataset']}: {rep['n_raw_rows']} dòng x {rep['n_raw_columns']} cột "
        f"(loại cột không phải feature: {rep['non_feature_columns']})",
        f"  Feature ban đầu: {rep['n_features_initial']} | bị loại: {rep['n_features_dropped']} | "
        f"còn lại đưa vào thuật toán: {rep['n_features_final']}",
    ]
    for d in rep["dropped_features"]:
        lines.append(f"    - {d['name']}: {d['reason']}")
    lines.append(f"  Dòng: còn {rep['n_rows_final']} (bị loại {rep['n_rows_dropped']}) | nhãn 1 = {rep['label_encoding']['1']!r}: "
                 f"{rep['n_positive']}, nhãn 0 = {rep['label_encoding']['0']!r}: {rep['n_negative']}")
    lines.append(f"  Ô NaN sẽ được impute (fit trên train): {rep['n_nan_cells']} | ô inf đổi thành NaN: {rep['n_inf_cells_to_nan']}")
    return "\n".join(lines)


def print_preprocess(rep):
    print(format_preprocess(rep))


def feature_frequency(results, dataset, feature_names):
    """Tỷ lệ các lần chạy (qua các seed) mà mỗi feature được chọn, theo từng thuật toán của một dataset."""
    groups = {}
    for r in results:
        if r["dataset"] == dataset:
            groups.setdefault(r["optimizer"], []).append(r["mask"])
    table = pd.DataFrame({k: np.mean(v, axis=0) for k, v in groups.items()}, index=feature_names)
    return table.assign(mean=table.mean(axis=1)).sort_values("mean", ascending=False)


# ----------------------------------------------------------------------------
# Biểu đồ
# ----------------------------------------------------------------------------
def _colors(cfg):
    return {o: f"C{i}" for i, o in enumerate(cfg["optimizers"])}


def plot_convergence(results, dataset, cfg, path):
    """Đường hội tụ: fitness tốt nhất (trung bình qua các seed ± std) theo số lần gọi fitness."""
    colors = _colors(cfg)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for opt in cfg["optimizers"]:
        histories = [r["history"] for r in results if r["dataset"] == dataset and r["optimizer"] == opt]
        if not histories:
            continue
        length = max(len(h) for h in histories)
        # Lần chạy dừng sớm: giữ nguyên giá trị tốt nhất cuối cùng cho đến hết trục x
        padded = np.array([h + [h[-1]] * (length - len(h)) for h in histories])
        mean, std = padded.mean(axis=0), padded.std(axis=0)
        x = np.arange(1, length + 1)
        ax.plot(x, mean, label=opt, color=colors[opt])
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=colors[opt])
    ax.set_xlabel("Số lần gọi fitness")
    ax.set_ylabel("Fitness tốt nhất (thấp hơn = tốt hơn)")
    ax.set_title(f"Đường hội tụ - {dataset} (trung bình ± std qua các seed)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(summary, stab, dataset, cfg, path):
    """4 biểu đồ: metric trên test, % feature giảm, thời gian (thực tế vs ước tính không cache), độ ổn định feature."""
    colors = _colors(cfg)
    primary = final_col(cfg["wrapper_model"], cfg["metric"])
    s = summary.loc[dataset]
    opts = [o for o in cfg["optimizers"] if o in s.index]
    x = np.arange(len(opts))
    c = [colors[o] for o in opts]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    ax = axes[0, 0]                                  # điểm + thanh lỗi (không bắt đầu từ 0 nên không dùng cột)
    ax.errorbar(x, s.loc[opts, f"{primary}_mean"], yerr=s.loc[opts, f"{primary}_std"], fmt="o", capsize=4, color="k")
    if BASELINE in s.index:
        ax.axhline(s.loc[BASELINE, f"{primary}_mean"], ls="--", color="gray", label="toàn bộ feature")
        ax.legend()
    ax.set_title(f"{primary} trên test (mean ± std qua seed)")

    axes[0, 1].bar(x, s.loc[opts, "reduction_pct_mean"], yerr=s.loc[opts, "reduction_pct_std"], capsize=4, color=c)
    axes[0, 1].set_title("% feature giảm được")

    ax = axes[1, 0]
    ax.bar(x - 0.2, s.loc[opts, "time_s_mean"], 0.4, label="thực tế (có cache)", color=c)
    ax.bar(x + 0.2, s.loc[opts, "time_nocache_s_mean"], 0.4, label="ước tính không cache", color=c, alpha=0.45)
    ax.set_title("Thời gian tìm kiếm (s)")
    ax.set_ylim(0, 1.3 * s.loc[opts, ["time_s_mean", "time_nocache_s_mean"]].to_numpy().max())   # chừa chỗ cho chú thích
    ax.legend(loc="upper center", ncol=2, fontsize=8)

    st = stab.loc[dataset]
    axes[1, 1].bar(x, st.loc[opts, "jaccard_mean"], color=c)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Độ ổn định tập feature giữa các seed (Jaccard)")

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(opts)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(dataset)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# In và xuất báo cáo
# ----------------------------------------------------------------------------
def _fmt(mean, std):
    if pd.isna(mean):
        return "-"
    return f"{mean:.4g}" if pd.isna(std) else f"{mean:.4g} ± {std:.2g}"


def _table_text(summary, dataset, columns):
    sub = summary.loc[dataset]
    table = pd.DataFrame(index=sub.index)
    for c in columns:
        table[c] = [_fmt(m, s) for m, s in zip(sub[f"{c}_mean"], sub[f"{c}_std"])]
    return table.to_string()


def build_report(summary, stab, cross, cfg, meta, consistency=""):
    """Ghép toàn bộ báo cáo thành một chuỗi."""
    groups = column_groups(cfg)
    lines = [
        f"Classifier wrapper: {cfg['wrapper_model']} | classifier đánh giá cuối: {final_models(cfg)} | "
        f"metric fitness: {cfg['metric']} | w = {cfg['w']}",
        f"Ngân sách: {cfg['max_evals']} ({cfg['budget_on']}) | patience: {cfg['patience']} | cache: {cfg['use_cache']} | "
        f"số seed: {len(cfg['seeds'])} | CV fitness: {cfg['cv_folds']}-fold | train cuối: {cfg['final_train_scope']}",
        consistency,
    ]
    for dataset in cfg["datasets"]:
        if dataset not in summary.index.get_level_values("dataset"):
            continue
        i = meta[dataset]["info"]
        sp = meta[dataset]["splits"]
        lines += ["", "=" * 100, format_preprocess(meta[dataset]["preprocess"]),
                  f"  Chia và xử lý theo seed ({len(sp)} seed): impute {cfg['impute_strategy']}, scale {cfg['scale']}, "
                  f"fit trên train rồi chỉ transform test; trong CV của fitness fit lại theo từng fold "
                  f"({cfg['cv_folds']} lần/seed); ô NaN đã impute ở train/test (seed đầu): "
                  f"{next(iter(sp.values()))['nan_imputed_train']}/{next(iter(sp.values()))['nan_imputed_test']}",
                  f"DATASET {dataset}: {i['n_samples']} mẫu ({i['phishing_rate']:.1%} phishing), {i['n_features']} feature | "
                  f"search {i['n_search']} / train cuối {i['n_final_train']} / test {i['n_test']}",
                  "(mean ± std qua các seed)"]
        for title, cols in groups.items():
            lines += ["", title, _table_text(summary, dataset, cols)]
        lines += ["", "4. TỔNG QUÁT HÓA QUA SEED (std của metric test, std số feature, Jaccard tập feature)",
                  stab.loc[dataset].round(4).to_string()]
    if len(cfg["datasets"]) > 1:
        lines += ["", "=" * 100, "5. TỔNG QUÁT HÓA QUA DATASET (hạng 1 = tốt nhất trong từng dataset)",
                  cross.round(4).to_string()]
    else:
        lines += ["", "(Chỉ chạy 1 dataset nên chưa có so sánh qua dataset.)"]
    return "\n".join(lines)


def save_report(results, baselines, cfg, meta, out_dir):
    """Tổng hợp kết quả, in báo cáo ra màn hình và lưu mọi thứ vào out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = results_to_df(results, baselines, cfg)
    groups = column_groups(cfg)
    all_cols = list(dict.fromkeys(c for cols in groups.values() for c in cols))
    summary = summarize(df, all_cols)
    stab = stability_table(summary, feature_stability(results), cfg)
    cross = cross_dataset_table(summary, cfg)

    df.to_csv(out_dir / "runs.csv", index=False)
    summary.to_csv(out_dir / "summary.csv")
    stab.to_csv(out_dir / "stability.csv")
    cross.to_csv(out_dir / "cross_dataset.csv")
    with open(out_dir / "selected_features.json", "w", encoding="utf-8") as f:
        names = {d: meta[d]["feature_names"] for d in meta}
        chosen = {f"{r['dataset']}|{r['optimizer']}|seed{r['seed']}":
                  [n for n, keep in zip(names[r["dataset"]], r["mask"]) if keep] for r in results}
        json.dump(chosen, f, ensure_ascii=False, indent=1)

    for dataset in meta:
        feature_frequency(results, dataset, meta[dataset]["feature_names"]).to_csv(
            out_dir / f"feature_frequency_{dataset}.csv")
        plot_convergence(results, dataset, cfg, out_dir / f"convergence_{dataset}.png")
        plot_comparison(summary, stab, dataset, cfg, out_dir / f"comparison_{dataset}.png")

    (out_dir / "preprocess_report.json").write_text(
        json.dumps({d: meta[d]["preprocess"] for d in meta}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    report = build_report(summary, stab, cross, cfg, meta, check_data_consistency(results, baselines)[1])
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nĐã lưu kết quả vào: {out_dir}")
    return summary


def report_from_dir(out_dir):
    """Tạo lại báo cáo từ raw_results.pkl trong một thư mục kết quả (để chỉnh evaluate mà không chạy lại thí nghiệm)."""
    with open(Path(out_dir) / "raw_results.pkl", "rb") as f:
        saved = pickle.load(f)
    return save_report(saved["results"], saved["baselines"], saved["cfg"], saved["meta"], out_dir)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if len(sys.argv) != 2:
        sys.exit("Cách dùng: python -m src.evaluate results/<thư mục kết quả>")
    report_from_dir(sys.argv[1])