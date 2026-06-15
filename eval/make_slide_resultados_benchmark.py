#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_slide_resultados_benchmark.py — Tabla maestra del benchmark Hito 2:
los 4 videos × 3 modelos (DPVO mono, DPVO métrico, MAC-VO) × métricas de
evaluación contra el GT de cintas (along-track).

Métricas: ATE [m], RPE traslacional 1 m [m], Deriva [%], Escala GT/est.
(RPE rotacional no calculable — GT sin orientación; fps va por modelo/plataforma.)

Números reales vía eval_ate_tape_gt.

Uso:
    .venv/bin/python scripts/make_slide_resultados_benchmark.py
Salida:
    docs/presentations/assets/fig_slide_resultados_benchmark.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from eval_ate_tape_gt import evaluate   # noqa: E402

ASSETS = REPO / "docs" / "presentations" / "assets"
NAVY = "#123B52"
METRIC_C = "#0072B2"
GOOD = "#1B7A3D"

GT = {"gym_v1": "results/gt/gym_video_1_gt_tum.txt",
      "gym_v2": "results/gt/gym_video_2_gt_tum.txt",
      "gym_v3": "results/gt/gym_video_3_gt_tum.txt",
      "video_4": "results/gt/underwater_video_4_gt_tum.txt"}
MONO = {"gym_v1": "results/2026-06-12__gym_v1_mono_r1__f0617bb9",
        "gym_v2": "results/2026-06-12__gym_v2_mono_r1__d0ad6a29",
        "gym_v3": "results/2026-06-12__gym_v3_mono_r1__59421744",
        "video_4": "results/2026-06-05__v4_off__fa45fd92"}
MET = {"gym_v1": "results/2026-06-09__gym_v1_metric_r1__9c41b983",
       "gym_v2": "results/2026-06-09__gym_v2_metric_r2__973c3764",
       "gym_v3": "results/2026-06-09__gym_v3_metric_r1__9f00aa88",
       "video_4": "results/2026-06-09__uw_v4_metric_r2__3b481569"}
MACVO = {"gym_v1": "results/macvo/MACVO-Fast@zed2i_gym_video1/06_10_151736",
         "gym_v2": "results/macvo/MACVO-Fast@zed2i_gym_video2/06_11_221250",
         "gym_v3": "results/macvo/MACVO-Fast@zed2i_gym_video3/06_11_223458",
         "video_4": "results/macvo/MACVO-Fast@zed2i_pool_video4/05_19_104541"}
VIDS = ["gym_v1", "gym_v2", "gym_v3", "video_4"]
MODELS = [("DPVO mono", MONO, "trajectory.txt"),
          ("DPVO métrico", MET, "trajectory.txt"),
          ("MAC-VO", MACVO, "trajectory_tum_mp4idx.txt")]


def main() -> None:
    cols = ["Video", "Modelo", "ATE [m]", "RPEₜ [m]", "Deriva [%]",
            "Escala GT/est"]
    rows, best_ate = [], {}
    for v in VIDS:
        ates = {}
        for name, d, f in MODELS:
            r = evaluate(REPO / GT[v], REPO / d[v] / f)
            ates[name] = r["ate_along_se1_m"]
            vname = v if name == "DPVO métrico" else ""   # video en fila del medio
            rows.append([vname, name, f"{r['ate_along_se1_m']:.3f}",
                         f"{r['rpe_t_1m_mean_m']:.3f}", f"{r['drift_pct']:.1f}",
                         f"{r['scale_gt_over_est']:.3f}"])
        best_ate[v] = min(ates, key=ates.get)

    fig, ax = plt.subplots(figsize=(15.5, 9.6), dpi=200)
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center",
                   bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False)
    tab.set_fontsize(16)

    for (rr, cc), cell in tab.get_celld().items():
        cell.set_edgecolor("white"); cell.set_linewidth(1.2)
        if rr == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", fontweight="bold")
            continue
        grp = (rr - 1) // 3
        within = (rr - 1) % 3
        v = VIDS[grp]
        model = MODELS[within][0]
        # fondo alterno por grupo de video; separador grueso arriba del grupo
        cell.set_facecolor("#EDF1F3" if grp % 2 == 0 else "#FFFFFF")
        if within == 0:
            cell.set_edgecolor("white")
            cell.visible_edges = "TLR" if cc else "TLR"
        if cc == 0:   # nombre del video (solo fila del medio), en negrita
            cell.set_text_props(fontweight="bold", color=NAVY)
        if cc == 1 and model == "DPVO métrico":
            cell.set_text_props(fontweight="bold", color=METRIC_C)
        if cc == 2 and model == best_ate[v]:   # mejor ATE del grupo
            cell.set_facecolor("#E2F0E8")
            cell.set_text_props(fontweight="bold", color=GOOD)

    # líneas separadoras entre grupos de video
    for g in range(1, 4):
        y = 1 - g * 3 / 12
        ax.plot([0, 1], [y, y], transform=ax.transAxes, color=NAVY, lw=1.8,
                zorder=10, clip_on=False)

    fig.suptitle("Resultados del benchmark — métricas vs GT de cintas",
                 fontsize=22, color=NAVY, fontweight="bold", y=0.985)
    fig.text(0.5, 0.022,
             "ATE y RPE traslacional medidos along-track (GT de cintas) · "
             "verde = mejor ATE por video · DPVO métrico = nuestro modelo",
             ha="center", fontsize=11.5, color="0.35")
    fig.text(0.5, -0.002,
             "el ATE alto del DPVO mono refleja su escala arbitraria (col. Escala ≠ 1), "
             "no error de forma · RPE rotacional no calculable (GT sin orientación) · "
             "fps por modelo/plataforma",
             ha="center", fontsize=10, color="0.45", style="italic")
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.07)
    out = ASSETS / "fig_slide_resultados_benchmark.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"OK {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
