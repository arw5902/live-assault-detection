"""
PR + ROC curves with threshold markers, F1 iso-curves, and summary tables.
Shared between src/train.py and src/evaluate.py.
"""

import math
import os
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)


_MARKER_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def _marker_metrics(y_true, y_score, thresholds):
    n_neg = max(1, int((y_true == 0).sum()))
    out = {}
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0)
        fp_count = int(((y_pred == 1) & (y_true == 0)).sum())
        out[t] = (p, r, f, fp_count / n_neg)
    return out, n_neg


def _fan_offsets(points, start_angle_deg, default_offset):
    """Radial spread for clustered label positions; falls back to default."""
    n = len(points)
    if n == 0:
        return []
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    clustered = (max(xs) - min(xs) < 0.05) and (max(ys) - min(ys) < 0.05)
    if clustered and n > 1:
        sweep = min(200, 30 * n)
        offs = []
        for i in range(n):
            angle = math.radians(start_angle_deg - (sweep * i / max(1, n - 1)))
            offs.append((18 * math.cos(angle), 18 * math.sin(angle)))
        return offs
    return [default_offset] * n


def plot_pr_and_roc(y_true, y_score, selected_threshold, model_type,
                    title_dataset, pr_path, roc_path):
    """
    Render PR + ROC curves and print a threshold-comparison table.

    Raises ImportError if matplotlib isn't installed — caller decides whether
    to swallow it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(pr_path) or ".", exist_ok=True)
    marker_m, n_neg = _marker_metrics(y_true, y_score, _MARKER_THRESHOLDS)

    # Make sure the selected threshold has an entry, even if not on the grid
    all_markers = list(_MARKER_THRESHOLDS)
    if not any(abs(t - selected_threshold) < 0.005 for t in _MARKER_THRESHOLDS):
        all_markers.append(selected_threshold)
        all_markers.sort()
        extra, _ = _marker_metrics(y_true, y_score, [selected_threshold])
        marker_m[selected_threshold] = extra[selected_threshold]

    # ── PR curve ────────────────────────────────────────────────────────────
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(8, 6))
    for f1_val in [0.5, 0.6, 0.7, 0.8, 0.9]:
        r_iso = np.linspace(0.01, 1.0, 200)
        p_iso = (f1_val * r_iso) / (2 * r_iso - f1_val)
        valid = (p_iso > 0) & (p_iso <= 1)
        ax.plot(r_iso[valid], p_iso[valid], '--', color='gray',
                alpha=0.35, linewidth=0.8)
        idx = np.where(valid)[0]
        if len(idx) > 0:
            li = idx[-1]
            ax.annotate(f"F1={f1_val}", (r_iso[li], p_iso[li]),
                        fontsize=7, color='gray', alpha=0.7,
                        ha='left', va='bottom')

    ax.plot(rec_arr, prec_arr, linewidth=2, color='#1f77b4',
            label=f"PR curve (AP={ap:.3f})")

    pr_markers = []
    for t in all_markers:
        p_m, r_m, f_m, fpr_m = marker_m[t]
        if r_m == 0 and p_m == 0:
            continue
        is_sel = abs(t - selected_threshold) < 0.005
        color = 'red' if is_sel else '#1f77b4'
        ax.plot(r_m, p_m, 'o', color=color,
                markersize=10 if is_sel else 6,
                zorder=10 if is_sel else 5)
        pr_markers.append((t, r_m, p_m, f_m, fpr_m, is_sel, color))

    pr_offsets = _fan_offsets([(m[1], m[2]) for m in pr_markers],
                              start_angle_deg=200, default_offset=(6, 4))

    table_lines = []
    for (t, r_m, p_m, f_m, fpr_m, is_sel, color), (dx, dy) in zip(pr_markers, pr_offsets):
        ax.annotate(f"{t:.2f}", (r_m, p_m),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=5.5, color=color,
                    fontweight='bold' if is_sel else 'normal',
                    arrowprops=dict(arrowstyle='-', color=color,
                                    lw=0.5, alpha=0.4)
                    if (abs(dx) > 10 or abs(dy) > 10) else None)
        tag = "★" if is_sel else " "
        table_lines.append(
            f"{tag} t={t:.2f}  P={p_m:.2f}  R={r_m:.2f}  "
            f"F1={f_m:.2f}  FPR={fpr_m:>6.2%}")

    txt = ax.text(0.02, 0.02, "\n".join(table_lines),
                  transform=ax.transAxes, fontsize=7,
                  fontfamily='monospace', verticalalignment='bottom',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor='#cccccc', alpha=0.9))
    # Render once so the legend can sit just above the table without overlap.
    fig.canvas.draw()
    bb = txt.get_window_extent(renderer=fig.canvas.get_renderer())
    bb_axes = bb.transformed(ax.transAxes.inverted())
    ax.legend(loc="lower left", fontsize=8,
              bbox_to_anchor=(0.01, bb_axes.y1 + 0.01))

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(f"Precision-Recall Curve ({title_dataset}) - {model_type.upper()}",
                 fontsize=12)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PR curve saved to  : {pr_path}  (AP={ap:.3f})")

    # ── ROC curve ───────────────────────────────────────────────────────────
    fpr_arr, tpr_arr, roc_thresh = roc_curve(y_true, y_score)
    roc_auc = auc(fpr_arr, tpr_arr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_arr, tpr_arr, linewidth=2, color='#1f77b4',
            label=f"ROC curve (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")

    roc_markers = []
    for t in all_markers:
        p_m, r_m, f_m, fpr_m = marker_m[t]
        # Snap to the nearest point on the actual ROC curve for the marker.
        best_dist = float('inf')
        tpr_m, fpr_plot = r_m, fpr_m
        for rt, rfp, rtp in zip(roc_thresh, fpr_arr, tpr_arr):
            d = abs(rt - t)
            if d < best_dist:
                best_dist = d
                fpr_plot, tpr_m = rfp, rtp
        is_sel = abs(t - selected_threshold) < 0.005
        color = 'red' if is_sel else '#1f77b4'
        ax.plot(fpr_plot, tpr_m, 'o', color=color,
                markersize=6 if is_sel else 3,
                zorder=10 if is_sel else 5)
        roc_markers.append((t, fpr_plot, tpr_m, is_sel, color, r_m, f_m, fpr_m))

    roc_offsets = _fan_offsets([(m[1], m[2]) for m in roc_markers],
                               start_angle_deg=-30, default_offset=(10, 0))

    roc_table_lines = []
    for (t, fpr_plot, tpr_m, is_sel, color, r_m, f_m, fpr_m), (dx, dy) in zip(roc_markers, roc_offsets):
        ax.annotate(f"{t:.2f}", (fpr_plot, tpr_m),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=5.5, color=color,
                    fontweight='bold' if is_sel else 'normal',
                    va='center',
                    arrowprops=dict(arrowstyle='-', color=color,
                                    lw=0.5, alpha=0.4)
                    if (abs(dx) > 10 or abs(dy) > 10) else None)
        tag = "★" if is_sel else " "
        roc_table_lines.append(
            f"{tag} t={t:.2f}  TPR={r_m:.2f}  FPR={fpr_m:>6.2%}  F1={f_m:.2f}")

    txt2 = ax.text(0.98, 0.02, "\n".join(roc_table_lines),
                   transform=ax.transAxes, fontsize=7,
                   fontfamily='monospace', verticalalignment='bottom',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                             edgecolor='#cccccc', alpha=0.9))
    fig.canvas.draw()
    bb2 = txt2.get_window_extent(renderer=fig.canvas.get_renderer())
    bb2_axes = bb2.transformed(ax.transAxes.inverted())
    ax.legend(loc="lower right", fontsize=8,
              bbox_to_anchor=(0.99, bb2_axes.y1 + 0.01))

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title(f"ROC Curve ({title_dataset}) - {model_type.upper()}", fontsize=12)
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curve saved to : {roc_path}  (AUC={roc_auc:.3f})")

    # ── Threshold comparison table ──────────────────────────────────────────
    print(f"\n  Threshold comparison (n_neg={n_neg}):")
    print(f"  {'thresh':>6}  {'Prec':>5}  {'Rec':>5}  {'F1':>5}  {'FP%':>6}  note")
    print(f"  {'-'*46}")
    for t in sorted(marker_m.keys()):
        p_m, r_m, f_m, fpr_m = marker_m[t]
        note = "★ selected" if abs(t - selected_threshold) < 0.005 else ""
        print(f"  {t:>6.2f}  {p_m:>5.3f}  {r_m:>5.3f}  {f_m:>5.3f}  {fpr_m:>6.2%}  {note}")
