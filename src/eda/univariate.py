"""
univariate.py - Framework phân tích Univariate cho các cột Continuous/Count.

Cách dùng nhanh (trong notebook):
    import sys; sys.path.append("..")
    from src.univariate import UnivariateAnalyzer

    ua = UnivariateAnalyzer(df, cols=continuous_cols)   # cols = list 50 cột liên tục/đếm
    ua.summary()                                        # bảng tổng quan 1 dòng / cột
    ua.show(["domain_age", "length_url"], ncols=2)      # báo cáo dạng thẻ xếp NGANG
    ua.report(["domain_age", "length_url"])             # BẢNG NGANG: dòng = cột dữ liệu
    ua.report(["domain_age"], layout="list")            # dạng danh sách dọc (cũ)
    ua.plot(["domain_age", "length_url"], ncols=4)      # Hist+KDE+Boxplot, xếp ngang
    ua.export_txt("reports/univariate_report.txt")      # xuất txt (mặc định: tất cả cột)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --------------------------------------------------------------------------- #
# Ngưỡng cấu hình (đổi ở đây nếu nhóm thống nhất ngưỡng khác)
# --------------------------------------------------------------------------- #
SKEW_MODERATE = 0.5      # 0.5 <= |skew| < 1  -> lệch vừa
SKEW_HIGH = 1.0          # |skew| >= 1        -> lệch mạnh
OUTLIER_PCT_WARN = 5.0   # % outlier (IQR) vượt ngưỡng này -> cảnh báo
ZERO_PCT_WARN = 50.0     # % giá trị 0 vượt ngưỡng này -> zero-inflated
LOW_CARD_MAX = 10        # số giá trị unique <= ngưỡng -> coi là rời rạc/ít mức
IQR_K = 1.5              # hệ số hàng rào Tukey


def skew_label(s: float) -> str:
    if np.isnan(s):
        return "N/A"
    a = abs(s)
    side = "phải (right)" if s > 0 else "trái (left)"
    if a < SKEW_MODERATE:
        return "xấp xỉ đối xứng"
    if a < SKEW_HIGH:
        return f"lệch vừa {side}"
    return f"lệch mạnh {side}"


class UnivariateAnalyzer:
    """Phân tích univariate có thể chọn cột tuỳ ý."""

    def __init__(self, df: pd.DataFrame, cols: Sequence[str] | None = None):
        if cols is None:
            cols = df.select_dtypes("number").columns.tolist()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Cột không tồn tại trong df: {missing}")
        self.df = df
        self.cols = list(cols)
        self._cache: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # Core: tính toàn bộ chỉ số cho 1 cột (có cache)
    # ------------------------------------------------------------------ #
    def _stats(self, col: str) -> dict:
        if col in self._cache:
            return self._cache[col]

        raw = self.df[col]
        x = raw.astype(float)
        finite = x[np.isfinite(x)]            # bỏ NaN, +inf, -inf để tính thống kê

        q1, med, q3 = finite.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - IQR_K * iqr, q3 + IQR_K * iqr
        n_low = int((finite < lo).sum())
        n_high = int((finite > hi).sum())
        n_out = n_low + n_high
        n = len(finite)
        skew = float(finite.skew()) if n > 2 and finite.nunique() > 1 else np.nan
        kurt = float(finite.kurt()) if n > 3 and finite.nunique() > 1 else np.nan

        s = dict(
            # --- Data validity ---
            n_nan=int(x.isna().sum()),
            n_posinf=int(np.isposinf(x).sum()),
            n_neginf=int(np.isneginf(x).sum()),
            n_neg=int((finite < 0).sum()),
            n_zero=int((finite == 0).sum()),
            # --- Descriptive ---
            count=int(finite.count()),
            mean=finite.mean(),
            std=finite.std(),
            min=finite.min(),
            q1=q1,
            median=med,
            q3=q3,
            max=finite.max(),
            nunique=int(finite.nunique()),
            # --- Distribution ---
            skew=skew,
            kurt=kurt,
            # --- Outlier (Tukey IQR) ---
            iqr=iqr,
            lower_fence=lo,
            upper_fence=hi,
            n_out_low=n_low,
            n_out_high=n_high,
            n_out=n_out,
            pct_out=100 * n_out / n if n else np.nan,
        )
        s["pct_zero"] = 100 * s["n_zero"] / n if n else np.nan
        s["pct_neg"] = 100 * s["n_neg"] / n if n else np.nan
        s["flags"] = self._flags(s)
        self._cache[col] = s
        return s

    @staticmethod
    def _flags(s: dict) -> list[str]:
        """Cờ cảnh báo (chỉ là gợi ý để DA kiểm tra, KHÔNG phải kết luận xử lý)."""
        f = []
        if s["n_posinf"] or s["n_neginf"]:
            f.append("INF_VALUE")
        if s["n_nan"]:
            f.append("HAS_NAN")
        if s["n_neg"]:
            f.append("NEGATIVE_VALUE")
        if s["nunique"] <= 1:
            f.append("CONSTANT")
        elif s["nunique"] <= LOW_CARD_MAX:
            f.append("LOW_CARDINALITY")
        if s["pct_zero"] >= ZERO_PCT_WARN:
            f.append("ZERO_INFLATED")
        if not np.isnan(s["skew"]) and abs(s["skew"]) >= SKEW_HIGH:
            f.append("HIGH_SKEW")
        elif not np.isnan(s["skew"]) and abs(s["skew"]) >= SKEW_MODERATE:
            f.append("MODERATE_SKEW")
        if s["pct_out"] >= OUTLIER_PCT_WARN:
            f.append("MANY_OUTLIERS")
        return f

    # ------------------------------------------------------------------ #
    # 1) Bảng tổng quan: 1 dòng / cột (để chọn cột cần đào sâu)
    # ------------------------------------------------------------------ #
    def summary(self, cols: Iterable[str] | None = None) -> pd.DataFrame:
        cols = list(cols) if cols is not None else self.cols
        rows = []
        for c in cols:
            s = self._stats(c)
            rows.append(
                {
                    "feature": c,
                    "nunique": s["nunique"],
                    "min": s["min"],
                    "median": s["median"],
                    "mean": s["mean"],
                    "max": s["max"],
                    "n_neg": s["n_neg"],
                    "n_inf": s["n_posinf"] + s["n_neginf"],
                    "pct_zero": round(s["pct_zero"], 1),
                    "skew": round(s["skew"], 2) if not np.isnan(s["skew"]) else np.nan,
                    "pct_outlier": round(s["pct_out"], 1),
                    "flags": ", ".join(s["flags"]),
                }
            )
        return pd.DataFrame(rows).set_index("feature")

    def filter_by_flag(self, flag: str) -> list[str]:
        """Lấy danh sách cột có cờ nào đó. VD: ua.filter_by_flag('HIGH_SKEW')."""
        return [c for c in self.cols if flag in self._stats(c)["flags"]]

    def top(self, by: str = "skew", n: int = 10, absolute: bool = True) -> list[str]:
        """Top n cột theo 'skew' | 'pct_out' | 'pct_zero'."""
        key = {"skew": "skew", "pct_out": "pct_out", "pct_zero": "pct_zero"}[by]
        vals = {c: self._stats(c)[key] for c in self.cols}
        vals = {c: (abs(v) if absolute else v) for c, v in vals.items() if not np.isnan(v)}
        return sorted(vals, key=vals.get, reverse=True)[:n]

    # ------------------------------------------------------------------ #
    # 2) Báo cáo dạng danh sách (text)
    # ------------------------------------------------------------------ #
    def _fmt(self, v) -> str:
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "NaN"
            return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:,.2f}"
        return str(v)

    def column_report(self, col: str) -> str:
        s, f = self._stats(col), self._fmt
        n_total = len(self.df)
        lines = [
            f"[{col}]",
            "  1. Data validity",
            f"     - +inf: {s['n_posinf']} | -inf: {s['n_neginf']} | NaN: {s['n_nan']}",
            f"     - Giá trị âm: {s['n_neg']} ({s['pct_neg']:.2f}%)"
            + (f" -> min = {f(s['min'])}" if s["n_neg"] else ""),
            "  2. Descriptive statistics",
            f"     - count : {f(s['count'])} / {n_total:,}",
            f"     - mean  : {f(s['mean'])}",
            f"     - std   : {f(s['std'])}",
            f"     - min   : {f(s['min'])}",
            f"     - Q1    : {f(s['q1'])}",
            f"     - median: {f(s['median'])}",
            f"     - Q3    : {f(s['q3'])}",
            f"     - max   : {f(s['max'])}",
            f"     - nunique: {s['nunique']:,} | zero: {s['n_zero']:,} ({s['pct_zero']:.1f}%)",
            "  3. Distribution",
            f"     - skewness: {f(s['skew'])} -> {skew_label(s['skew'])}",
            f"     - kurtosis (excess): {f(s['kurt'])}",
            "  4. Outlier (Tukey IQR, k=%.1f)" % IQR_K,
            f"     - IQR = {f(s['iqr'])} | hàng rào: [{f(s['lower_fence'])}, {f(s['upper_fence'])}]",
            f"     - Outlier: {s['n_out']:,} ({s['pct_out']:.2f}%) "
            f"[thấp: {s['n_out_low']:,} | cao: {s['n_out_high']:,}]",
            "  5. Flags: " + (", ".join(s["flags"]) if s["flags"] else "OK"),
        ]
        return "\n".join(lines)

    def table(self, cols: Iterable[str] | None = None, precision: int = 3) -> pd.DataFrame:
        """
        Bảng NGANG: mỗi DÒNG = 1 cột dữ liệu, mỗi CỘT = 1 chỉ số,
        gom theo nhóm Validity / Descriptive / Distribution / Outlier.
        """
        cols = list(cols) if cols is not None else self.cols
        rows = {}
        for c in cols:
            s = self._stats(c)
            rows[c] = {
                ("Validity", "+inf"): s["n_posinf"],
                ("Validity", "-inf"): s["n_neginf"],
                ("Validity", "NaN"): s["n_nan"],
                ("Validity", "n_neg"): s["n_neg"],
                ("Validity", "%neg"): s["pct_neg"],
                ("Descriptive", "count"): s["count"],
                ("Descriptive", "mean"): s["mean"],
                ("Descriptive", "std"): s["std"],
                ("Descriptive", "min"): s["min"],
                ("Descriptive", "Q1"): s["q1"],
                ("Descriptive", "median"): s["median"],
                ("Descriptive", "Q3"): s["q3"],
                ("Descriptive", "max"): s["max"],
                ("Descriptive", "nunique"): s["nunique"],
                ("Descriptive", "%zero"): s["pct_zero"],
                ("Distribution", "skew"): s["skew"],
                ("Distribution", "kurt"): s["kurt"],
                ("Distribution", "shape"): skew_label(s["skew"]),
                ("Outlier", "n_out"): s["n_out"],
                ("Outlier", "%out"): s["pct_out"],
                ("Flags", "flags"): ", ".join(s["flags"]),
            }
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df.index.name = "feature"
        num_cols = df.select_dtypes("number").columns
        df[num_cols] = df[num_cols].round(precision)
        return df

    def report(self, cols: Iterable[str] | None = None, layout: str = "table",
               print_out: bool = True):
        """
        layout="table": bảng ngang (dòng = cột dữ liệu, cột = chỉ số) -> trả về DataFrame,
                        tự hiển thị trong notebook nếu print_out=True.
        layout="list" : dạng danh sách dọc cũ (text) -> trả về str.
        """
        cols = list(cols) if cols is not None else self.cols
        if layout == "list":
            text = "\n\n".join(self.column_report(c) for c in cols)
            if print_out:
                print(text)
            return text
        tb = self.table(cols)
        if print_out:
            try:
                from IPython.display import display
                with pd.option_context("display.max_columns", None, "display.width", None,
                                       "display.max_colwidth", None, "display.max_rows", None):
                    display(tb)
            except ImportError:
                print(tb.to_string())
        return tb

    # ------------------------------------------------------------------ #
    # 3) Biểu đồ: Hist + KDE (trái) và Boxplot (phải), 1 hàng / cột
    # ------------------------------------------------------------------ #
    def plot(
        self,
        cols: Iterable[str] | None = None,
        layout: str = "horizontal",
        ncols: int = 4,
        per_fig: int = 4,
        clip_upper_q: float | None = None,
        save_dir: str | Path | None = None,
        show: bool = True,
        color: str = "#4C72B0",
    ):
        """
        layout      : "horizontal" -> mỗi cột là 1 khung dọc (Hist+KDE trên, Boxplot dưới,
                      chung trục x), các cột xếp NGANG, `ncols` cột / hàng.
                      "vertical"   -> bố cục cũ: mỗi cột 1 hàng (Hist trái, Boxplot phải),
                      `per_fig` cột / hình.
        clip_upper_q: vd 0.99 -> Hist+KDE chỉ vẽ đến phân vị 99% để thấy rõ thân phân phối
                      khi đuôi quá dài (Boxplot và mọi chỉ số vẫn tính trên dữ liệu đầy đủ;
                      lưu ý ở layout ngang hai biểu đồ chung trục x nên Boxplot cũng bị cắt
                      hiển thị cùng khoảng).
        save_dir    : nếu có, lưu mỗi figure thành PNG.
        """
        cols = list(cols) if cols is not None else self.cols
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        horizontal = layout == "horizontal"
        step = ncols if horizontal else per_fig

        figs = []
        for i in range(0, len(cols), step):
            chunk = cols[i : i + step]
            if horizontal:
                fig, axes = plt.subplots(
                    2, len(chunk), figsize=(4.2 * len(chunk), 5.2), squeeze=False,
                    gridspec_kw={"height_ratios": [3, 1]}, sharex="col",
                )
            else:
                fig, axes = plt.subplots(
                    len(chunk), 2, figsize=(12, 3 * len(chunk)), squeeze=False,
                    gridspec_kw={"width_ratios": [2, 1]},
                )
            for r, c in enumerate(chunk):
                s = self._stats(c)
                x = self.df[c].astype(float)
                x = x[np.isfinite(x)]
                x_full = x                      # boxplot luôn dùng dữ liệu đầy đủ
                note = ""
                if clip_upper_q is not None:    # chỉ hist+KDE bị cắt đuôi
                    cap = x.quantile(clip_upper_q)
                    x = x[x <= cap]
                    note = f" (≤P{int(clip_upper_q * 100)})"

                ax_h, ax_b = (axes[0, r], axes[1, r]) if horizontal else axes[r]
                if s["nunique"] <= 1:
                    ax_h.text(0.5, 0.5, f"Hằng số = {self._fmt(s['min'])}",
                              ha="center", va="center", transform=ax_h.transAxes)
                    ax_b.axis("off")
                else:
                    discrete = s["nunique"] <= LOW_CARD_MAX
                    sns.histplot(x, kde=not discrete, discrete=discrete, ax=ax_h,
                                 color=color, edgecolor="white")
                    ax_h.axvline(s["mean"], color="crimson", ls="--", lw=1, label="mean")
                    ax_h.axvline(s["median"], color="green", ls="-", lw=1, label="median")
                    ax_h.legend(fontsize=8)
                    sns.boxplot(x=x_full, ax=ax_b, color=color, fliersize=2)
                    if horizontal and clip_upper_q is not None:
                        pad = 0.03 * (x.max() - x.min() or 1)
                        ax_h.set_xlim(x.min() - pad, x.max() + pad)   # sharex -> áp cho cả boxplot

                skew_txt = f"skew={self._fmt(s['skew'])} ({skew_label(s['skew'])})"
                if horizontal:
                    ax_h.set_title(f"{c}{note}\n{skew_txt}", fontsize=10)
                    ax_b.set_title("")
                    ax_b.set_xlabel(f"Outlier: {s['n_out']:,} ({s['pct_out']:.1f}%)", fontsize=9)
                    if r > 0:
                        ax_h.set_ylabel("")
                else:
                    ax_h.set_title(f"{c}{note} | {skew_txt}", fontsize=10, loc="left")
                    ax_b.set_title(f"Outlier IQR: {s['n_out']:,} ({s['pct_out']:.1f}%)", fontsize=10)
                    ax_b.set_xlabel("")
                if not horizontal:
                    ax_h.set_xlabel("")
            fig.tight_layout()
            if save_dir:
                fig.savefig(Path(save_dir) / f"univariate_{i // step + 1:02d}.png",
                            dpi=130, bbox_inches="tight")
            figs.append(fig)
            if show:
                plt.show()
            else:
                plt.close(fig)
        return figs

    def show(self, cols: Iterable[str] | None = None, ncols: int = 4, font_px: int = 12):
        """Hiển thị báo cáo dạng danh sách theo các thẻ xếp NGANG trong notebook."""
        import html
        from IPython.display import HTML, display

        cols = list(cols) if cols is not None else self.cols
        cards = "".join(
            f'<div style="flex:1 1 {100 // ncols - 2}%;min-width:240px;border:1px solid #8884;'
            f'border-radius:6px;padding:6px 10px;margin:4px;">'
            f'<pre style="margin:0;font-size:{font_px}px;line-height:1.35;white-space:pre-wrap;">'
            f"{html.escape(self.column_report(c))}</pre></div>"
            for c in cols
        )
        display(HTML(f'<div style="display:flex;flex-wrap:wrap;">{cards}</div>'))

    