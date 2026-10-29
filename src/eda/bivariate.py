"""
bivariate.py - Framework phân tích Bivariate (target nhị phân).

PHẦN A - Feature vs Target
    ba.target_table()          bảng ngang: dòng = feature liên tục, cột = chỉ số tách lớp
    ba.plot_target(cols)       KDE theo lớp (trên) + Boxplot theo lớp (dưới), xếp ngang
    ba.plot_target_rank()      xếp hạng độ tách lớp của toàn bộ feature trong 1 hình
    ba.binary_target_table()   bảng ngang cho feature nhị phân
    ba.plot_binary_target()    P(x=1 | lớp) của toàn bộ feature nhị phân trong 1 hình

PHẦN B - Feature vs Feature
    ba.corr_pairs(thr)         các cặp tương quan cao (kèm độ tách lớp của từng cột)
    ba.redundant_groups(thr)   gom các cột tương quan cao thành nhóm dư thừa
    ba.plot_corr()             heatmap tương quan (gom cụm)
    ba.plot_scatter(pairs)     scatter từng cặp, tô màu theo target, xếp ngang
"""
from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# --------------------------------------------------------------------------- #
# Ngưỡng cấu hình
# --------------------------------------------------------------------------- #
SEP_STRONG = 0.5     # sep = |2*AUC - 1| >= 0.5  -> tách lớp mạnh
SEP_WEAK = 0.1       # sep < 0.1                 -> gần như không tách lớp
LOW_CARD_MAX = 10    # nunique <= ngưỡng -> vẽ rời rạc (không KDE)
PHI_STRONG = 0.5     # |phi| >= 0.5 (binary vs target)
PHI_WEAK = 0.05      # |phi| < 0.05
RARE_PCT = 1.0       # % giá trị 1 (hoặc 0) < ngưỡng -> hiếm
COLORS = {0: "#4C72B0", 1: "#DD8452"}


class BivariateAnalyzer:
    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        cont_cols: Sequence[str],
        bin_cols: Sequence[str] | None = None,
        positive=None,
    ):
        """
        target   : tên cột target (2 lớp).
        positive : nhãn lớp dương (mã hoá 1). Mặc định: giá trị lớn nhất theo thứ tự sắp xếp.
        """
        classes = sorted(df[target].dropna().unique().tolist())
        if len(classes) != 2:
            raise ValueError(f"Target phải có đúng 2 lớp, hiện có: {classes}")
        if positive is None:
            positive = classes[-1]
        if positive not in classes:
            raise ValueError(f"positive={positive!r} không thuộc {classes}")
        negative = classes[0] if classes[1] == positive else classes[1]

        self.df = df
        self.target = target
        self.cont_cols = list(cont_cols)
        self.bin_cols = list(bin_cols) if bin_cols is not None else []
        self.y = (df[target] == positive).astype(int)
        self.lbl = {0: str(negative), 1: str(positive)}
        self.order = [self.lbl[0], self.lbl[1]]
        self.palette = {self.lbl[0]: COLORS[0], self.lbl[1]: COLORS[1]}
        self._cache: dict[str, dict] = {}
        print(f"Target='{target}': 0 = {self.lbl[0]} ({(self.y == 0).sum():,}) | "
              f"1 = {self.lbl[1]} ({(self.y == 1).sum():,})")

    # ====================================================================== #
    # PHẦN A - FEATURE vs TARGET
    # ====================================================================== #
    def _cont_stats(self, col: str) -> dict:
        if col in self._cache:
            return self._cache[col]
        x = self.df[col].astype(float)
        ok = np.isfinite(x)
        x, y = x[ok], self.y[ok]
        x1, x0 = x[y == 1], x[y == 0]

        auc = p_mwu = ks = rho = np.nan
        if x.nunique() > 1:
            u, p_mwu = stats.mannwhitneyu(x1, x0, alternative="two-sided")
            auc = u / (len(x1) * len(x0))          # AUC đơn biến (P[x lớp 1 > x lớp 0])
            ks = stats.ks_2samp(x1, x0, method="asymp").statistic
            rho = stats.spearmanr(x, y)[0]
        sep = abs(2 * auc - 1) if not np.isnan(auc) else np.nan

        flags = []
        if x.nunique() <= 1:
            flags.append("CONSTANT")
        elif sep >= SEP_STRONG:
            flags.append("STRONG_SEP")
        elif sep < SEP_WEAK:
            flags.append("WEAK_SEP")
        if x.nunique() > 1 and np.isclose(x1.median(), x0.median()) and ks >= 0.2:
            flags.append("SAME_MEDIAN_DIFF_SHAPE")

        s = {
            f"mean[{self.lbl[0]}]": x0.mean(), f"mean[{self.lbl[1]}]": x1.mean(),
            f"median[{self.lbl[0]}]": x0.median(), f"median[{self.lbl[1]}]": x1.median(),
            "AUC": auc, "sep": sep,
            "dir": ("+" if auc > 0.5 else "-") if not np.isnan(auc) else "",
            "KS": ks, "spearman": rho, "p_mwu": p_mwu,
            "flags": ", ".join(flags),
        }
        self._cache[col] = s
        return s

    def target_table(self, cols: Iterable[str] | None = None, sort: bool = True,
                     precision: int = 3) -> pd.DataFrame:
        """
        Bảng ngang feature liên tục vs target.
        AUC : xác suất giá trị lớp 1 > lớp 0 (0.5 = không phân biệt).
        sep : |2*AUC-1| in [0,1] -> độ tách lớp (không phụ thuộc chiều).
        dir : '+' nếu lớp 1 có giá trị cao hơn, '-' nếu thấp hơn.
        KS  : khoảng cách lớn nhất giữa 2 hàm phân phối tích luỹ.
        """
        cols = list(cols) if cols is not None else self.cont_cols
        tb = pd.DataFrame({c: self._cont_stats(c) for c in cols}).T
        tb.index.name = "feature"
        num = [c for c in tb.columns if c not in ("dir", "flags", "p_mwu")]
        tb[num] = tb[num].astype(float).round(precision)
        tb["p_mwu"] = tb["p_mwu"].astype(float)
        if sort:
            tb = tb.sort_values("sep", ascending=False)
        return tb

    def top_target(self, n: int = 10, by: str = "sep") -> list[str]:
        """Top n feature liên tục theo 'sep' | 'KS'."""
        vals = {c: self._cont_stats(c)[by] for c in self.cont_cols}
        vals = {c: v for c, v in vals.items() if not np.isnan(v)}
        return sorted(vals, key=vals.get, reverse=True)[:n]

    def filter_by_flag(self, flag: str) -> list[str]:
        return [c for c in self.cont_cols if flag in self._cont_stats(c)["flags"]]

    def plot_target(self, cols: Iterable[str], ncols: int = 4,
                    clip_upper_q: float | None = 0.99, show: bool = True):
        """KDE theo lớp (trên) + Boxplot theo lớp (dưới); mỗi feature 1 khung, xếp ngang."""
        cols = list(cols)
        figs = []
        for i in range(0, len(cols), ncols):
            chunk = cols[i : i + ncols]
            fig, axes = plt.subplots(2, len(chunk), figsize=(4.2 * len(chunk), 5.2),
                                     squeeze=False, sharex="col",
                                     gridspec_kw={"height_ratios": [3, 1.3]})
            for r, c in enumerate(chunk):
                s = self._cont_stats(c)
                d = pd.DataFrame({"v": self.df[c].astype(float),
                                  "cls": self.y.map(self.lbl)})
                d = d[np.isfinite(d["v"])]
                dp = d
                note = ""
                if clip_upper_q is not None:
                    cap = d["v"].quantile(clip_upper_q)
                    dp = d[d["v"] <= cap]
                    note = f" (≤P{int(clip_upper_q * 100)})"
                ax_h, ax_b = axes[0, r], axes[1, r]

                if d["v"].nunique() <= 1:
                    ax_h.text(0.5, 0.5, "Hằng số", ha="center", va="center",
                              transform=ax_h.transAxes)
                    ax_b.axis("off")
                else:
                    if d["v"].nunique() <= LOW_CARD_MAX:
                        sns.histplot(data=dp, x="v", hue="cls", hue_order=self.order,
                                     palette=self.palette, discrete=True, stat="proportion",
                                     common_norm=False, multiple="dodge", shrink=0.8, ax=ax_h)
                    else:
                        sns.kdeplot(data=dp, x="v", hue="cls", hue_order=self.order,
                                    palette=self.palette, common_norm=False, fill=True,
                                    alpha=0.35, cut=0, ax=ax_h)
                    sns.boxplot(data=d, x="v", y="cls", order=self.order, hue="cls",
                                hue_order=self.order, palette=self.palette, legend=False,
                                fliersize=2, ax=ax_b)
                    if clip_upper_q is not None:
                        pad = 0.03 * ((dp["v"].max() - dp["v"].min()) or 1)
                        ax_h.set_xlim(dp["v"].min() - pad, dp["v"].max() + pad)
                    if ax_h.get_legend() is not None:
                        ax_h.get_legend().set_title(self.target)

                ax_h.set_title(f"{c}{note}\nAUC={s['AUC']:.2f} | sep={s['sep']:.2f} | KS={s['KS']:.2f}",
                               fontsize=10)
                ax_h.set_xlabel("")
                ax_b.set_xlabel("")
                ax_b.set_ylabel("")
                if r > 0:
                    ax_h.set_ylabel("")
                    ax_b.set_yticklabels([])
            fig.tight_layout()
            figs.append(fig)
            plt.show() if show else plt.close(fig)
        return figs

    def plot_target_rank(self, top: int | None = None, show: bool = True):
        """Xếp hạng sep (độ tách lớp) của các feature liên tục; màu theo chiều (+/-)."""
        tb = self.target_table()
        tb = tb.dropna(subset=["sep"])
        if top:
            tb = tb.head(top)
        tb = tb.iloc[::-1]
        colors = [COLORS[1] if d == "+" else COLORS[0] for d in tb["dir"]]
        fig, ax = plt.subplots(figsize=(8, 0.28 * len(tb) + 1.5))
        ax.barh(tb.index, tb["sep"], color=colors)
        ax.axvline(SEP_STRONG, color="green", ls="--", lw=0.8)
        ax.axvline(SEP_WEAK, color="red", ls="--", lw=0.8)
        ax.set(xlabel="sep = |2*AUC - 1|", xlim=(0, 1),
               title=f"Độ tách lớp theo '{self.target}' (cam: lớp '{self.lbl[1]}' cao hơn, "
                     f"xanh: thấp hơn)")
        fig.tight_layout()
        plt.show() if show else plt.close(fig)
        return fig

    # ---------------------------- binary vs target ------------------------- #
    def binary_target_table(self, cols: Iterable[str] | None = None,
                            precision: int = 3) -> pd.DataFrame:
        """
        Bảng ngang feature nhị phân (0/1) vs target.
        P(x=1|y=k): tỷ lệ x=1 trong từng lớp | P(y=1|x=k): tỷ lệ lớp dương theo giá trị x.
        phi: tương quan có dấu | OR: odds ratio (hiệu chỉnh +0.5).
        """
        cols = list(cols) if cols is not None else self.bin_cols
        rows = {}
        for c in cols:
            x = self.df[c]
            ct = pd.crosstab(x, self.y).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            n = ct.values.sum()
            n_x1 = int(ct.loc[1].sum())
            n_x0 = int(ct.loc[0].sum())
            phi = x.corr(self.y) if x.nunique() > 1 else np.nan
            if min(n_x0, n_x1) > 0 and ct.sum(axis=0).min() > 0:
                p_chi = stats.chi2_contingency(ct.values, correction=False)[1]
            else:
                p_chi = np.nan
            a, b, c_, d_ = ct.loc[1, 1] + .5, ct.loc[1, 0] + .5, ct.loc[0, 1] + .5, ct.loc[0, 0] + .5
            flags = []
            if x.nunique() <= 1:
                flags.append("CONSTANT")
            else:
                if min(n_x1, n_x0) / n * 100 < RARE_PCT:
                    flags.append("RARE")
                if abs(phi) >= PHI_STRONG:
                    flags.append("STRONG")
                elif abs(phi) < PHI_WEAK:
                    flags.append("WEAK")
            rows[c] = {
                "%x=1": 100 * n_x1 / n,
                f"P(x=1|{self.lbl[0]})": ct.loc[1, 0] / ct[0].sum() if ct[0].sum() else np.nan,
                f"P(x=1|{self.lbl[1]})": ct.loc[1, 1] / ct[1].sum() if ct[1].sum() else np.nan,
                "P(y=1|x=1)": ct.loc[1, 1] / n_x1 if n_x1 else np.nan,
                "P(y=1|x=0)": ct.loc[0, 1] / n_x0 if n_x0 else np.nan,
                "phi": phi, "OR": (a * d_) / (b * c_), "p_chi2": p_chi,
                "flags": ", ".join(flags),
            }
        tb = pd.DataFrame(rows).T
        tb.index.name = "feature"
        num = [c for c in tb.columns if c not in ("flags", "p_chi2")]
        tb[num] = tb[num].astype(float).round(precision)
        tb["p_chi2"] = tb["p_chi2"].astype(float)
        return tb.reindex(tb["phi"].abs().sort_values(ascending=False).index)

    def plot_binary_target(self, cols: Iterable[str] | None = None, show: bool = True):
        """So sánh P(x=1 | lớp) giữa 2 lớp cho toàn bộ feature nhị phân trong 1 hình."""
        tb = self.binary_target_table(cols)
        k0, k1 = f"P(x=1|{self.lbl[0]})", f"P(x=1|{self.lbl[1]})"
        tb = tb.iloc[::-1]
        pos = np.arange(len(tb))
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(tb) + 1.5))
        h = 0.38
        ax.barh(pos + h / 2, tb[k0] * 100, h, color=COLORS[0], label=self.lbl[0])
        ax.barh(pos - h / 2, tb[k1] * 100, h, color=COLORS[1], label=self.lbl[1])
        ax.set_yticks(pos, tb.index)
        ax.set(xlabel="% quan sát có x = 1 trong từng lớp", xlim=(0, 100),
               title="P(x=1 | lớp) của các feature nhị phân (sắp theo |phi|)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        plt.show() if show else plt.close(fig)
        return fig

    # ====================================================================== #
    # PHẦN B - FEATURE vs FEATURE
    # ====================================================================== #
    def corr_matrix(self, cols: Iterable[str] | None = None,
                    method: str = "spearman") -> pd.DataFrame:
        """Ma trận tương quan. Mặc định Spearman (dữ liệu lệch/outlier). Bỏ cột hằng số."""
        cols = list(cols) if cols is not None else self.cont_cols
        const = [c for c in cols if self.df[c].nunique() <= 1]
        if const:
            print(f"[corr] Bỏ cột hằng số: {const}")
        cols = [c for c in cols if c not in const]
        return self.df[cols].corr(method=method)

    def corr_pairs(self, cols: Iterable[str] | None = None, thr: float = 0.8,
                   method: str = "spearman", precision: int = 3) -> pd.DataFrame:
        """
        Các cặp có |tương quan| >= thr, kèm sep (độ tách lớp) của từng cột
        để biết cột nào mang nhiều thông tin target hơn trong cặp.
        """
        corr = self.corr_matrix(cols, method)
        low = corr.where(np.tril(np.ones(corr.shape, dtype=bool), -1)).stack()
        pr = low.rename("corr").reset_index()
        pr.columns = ["col_a", "col_b", "corr"]
        pr = pr[pr["corr"].abs() >= thr].copy()
        sep = lambda c: self._cont_stats(c)["sep"] if c in self.cont_cols else np.nan
        pr["sep_a"] = pr["col_a"].map(sep)
        pr["sep_b"] = pr["col_b"].map(sep)
        pr = pr.reindex(pr["corr"].abs().sort_values(ascending=False).index)
        return pr.round(precision).reset_index(drop=True)

    def redundant_groups(self, cols: Iterable[str] | None = None, thr: float = 0.8,
                         method: str = "spearman") -> pd.DataFrame:
        """Gom các cột liên thông qua cặp |corr| >= thr thành nhóm dư thừa."""
        pr = self.corr_pairs(cols, thr, method)
        adj: dict[str, set] = {}
        for a, b in zip(pr["col_a"], pr["col_b"]):
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        seen, groups = set(), []
        for node in adj:
            if node in seen:
                continue
            stack, comp = [node], []
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                seen.add(u)
                comp.append(u)
                stack.extend(adj[u] - seen)
            groups.append(sorted(comp))
        rows = []
        for g in sorted(groups, key=len, reverse=True):
            seps = {c: self._cont_stats(c)["sep"] for c in g if c in self.cont_cols}
            rows.append({"size": len(g), "features": ", ".join(g),
                         "highest_sep": max(seps, key=seps.get) if seps else None})
        out = pd.DataFrame(rows)
        out.index = range(1, len(out) + 1)
        out.index.name = "group"
        return out

    def plot_corr(self, cols: Iterable[str] | None = None, method: str = "spearman",
                  cluster: bool = True, annot: bool | None = None, show: bool = True):
        corr = self.corr_matrix(cols, method)
        if cluster:
            g = sns.clustermap(corr.fillna(0), method="average")
            order = g.dendrogram_row.reordered_ind
            plt.close(g.figure)
            corr = corr.iloc[order, order]
        if annot is None:
            annot = len(corr) <= 25
        n = len(corr)
        side = max(8, 0.32 * n + 3)
        fig, ax = plt.subplots(figsize=(side + 1, side))
        sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), cmap="coolwarm",
                    vmin=-1, vmax=1, center=0, annot=annot, fmt=".1f",
                    annot_kws={"size": 6}, linewidths=0.2, cbar_kws={"label": method}, ax=ax)
        ax.set_title(f"Ma trận tương quan {method}" + (" (gom cụm)" if cluster else ""))
        fig.tight_layout()
        plt.show() if show else plt.close(fig)
        return fig

    def plot_scatter(self, pairs: Sequence[tuple[str, str]], ncols: int = 3,
                     clip_upper_q: float | None = 0.99, sample: int | None = 3000,
                     method: str = "spearman", show: bool = True):
        """Scatter từng cặp (tô màu theo target), xếp ngang. pairs=[("a","b"), ...]"""
        pairs = list(pairs)
        figs = []
        for i in range(0, len(pairs), ncols):
            chunk = pairs[i : i + ncols]
            fig, axes = plt.subplots(1, len(chunk), figsize=(4.6 * len(chunk), 4.4),
                                     squeeze=False)
            for ax, (a, b) in zip(axes[0], chunk):
                d = pd.DataFrame({a: self.df[a].astype(float), b: self.df[b].astype(float),
                                  "cls": self.y.map(self.lbl)})
                d = d[np.isfinite(d[a]) & np.isfinite(d[b])]
                rho = d[[a, b]].corr(method=method).iloc[0, 1]
                if sample and len(d) > sample:
                    d = d.sample(sample, random_state=0)
                sns.scatterplot(data=d, x=a, y=b, hue="cls", hue_order=self.order,
                                palette=self.palette, s=10, alpha=0.4, linewidth=0, ax=ax)
                if clip_upper_q is not None:
                    ax.set_xlim(None, d[a].quantile(clip_upper_q))
                    ax.set_ylim(None, d[b].quantile(clip_upper_q))
                ax.set_title(f"{method} = {rho:.2f}", fontsize=10)
                ax.legend(title="", fontsize=8, markerscale=1.5)
            fig.tight_layout()
            figs.append(fig)
            plt.show() if show else plt.close(fig)
        return figs