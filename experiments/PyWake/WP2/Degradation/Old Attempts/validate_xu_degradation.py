# =============================================================================
# Validation against Xu et al. (2016) Fig. 5
# =============================================================================
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from degradation_xu import (
    XuModelParams,
    XU_LMO,
    fc_cycle,
    xu_capacity_curve,
)

def validate_xu_dst(
    save_path: Optional[str] = "xu_validation_dst.png",
    show: bool = True,
    T_cell_C: float = 21.0,
) -> "plt.Figure":
    """Reproduce Fig. 5 of Xu et al. (2016): model vs. experimental DST data.

    Produces a two-panel figure mirroring the paper's layout:
      Left ,  digitised experimental data from Fig. 5a.
      Right,  this implementation's model reproduction (Fig. 5b equivalent).

    Model assumptions
    -----------------
    Each DST test cycle is modelled as a SINGLE full rainflow cycle with:
      DoD  = soc_start − soc_stop  (the full swing from start to stop)
      σ    = (soc_start + soc_stop) / 2  (mean SoC)

    This is the correct minimal approximation: the discharge half and the
    1C recharge half form one complete rainflow cycle.  The DST profile
    does create small internal micro-cycles (visible in Fig. 4c of the paper),
    but without the proprietary DST signal those cannot be reproduced exactly.
    Omitting them is consistent with the paper's own note that their 14%
    error comes from "the absence of the original SoC profile".

    Temperature is fixed at 21°C as stated in Section V.B of the paper
    ("simulated average cell temperature was 21°C").

    Digitised reference data
    ------------------------
    ~10 points per curve manually extracted from Fig. 5a.  Used as scatter
    markers on the left panel so the model curves on the right can be
    compared directly against experiment.

    Args:
        save_path:  Path to save the output figure.  None = don't save.
        show:       If True, calls plt.show().
        T_cell_C:   Cell temperature [°C].  Paper uses 21°C.

    Returns:
        Matplotlib Figure object.
    """
    p = XU_LMO

    # ------------------------------------------------------------------
    # 7 DST test conditions from Fig. 5 legend: (label, soc_start, soc_stop)
    # ------------------------------------------------------------------
    dst_cases = [
        ("100-25@20C", 1.00, 0.25),   # DoD=75%, mean SoC=62.5% ,  most degradation
        ("100-40@20C", 1.00, 0.40),   # DoD=60%, mean SoC=70%
        ("85-25@20C",  0.85, 0.25),   # DoD=60%, mean SoC=55%
        ("100-50@20C", 1.00, 0.50),   # DoD=50%, mean SoC=75%
        ("75-25@20C",  0.75, 0.25),   # DoD=50%, mean SoC=50%
        ("75-45@20C",  0.75, 0.45),   # DoD=30%, mean SoC=60%
        ("75-65@20C",  0.75, 0.65),   # DoD=10%, mean SoC=70% ,  least degradation
    ]

    colours = ["black", "red", "green", "navy", "teal", "magenta", "goldenrod"]

    # ------------------------------------------------------------------
    # Digitised reference data,  Fig. 5a, Xu et al. (2016)
    # Manually extracted; ~10 points per curve.
    # ------------------------------------------------------------------
    fig5a_data = {
        "100-25@20C": [
            (0, 100.0), (300, 97.0), (700, 94.0), (1200, 91.5),
            (2000, 89.0), (3000, 86.5), (4500, 84.0), (6000, 82.0),
            (7500, 80.0), (9000, 78.5),
        ],
        "100-40@20C": [
            (0, 100.0), (400, 97.5), (900, 95.5), (1800, 93.0),
            (3000, 90.5), (4500, 88.0), (6000, 86.0), (7500, 84.0),
            (9000, 82.0),
        ],
        "85-25@20C": [
            (0, 100.0), (500, 98.0), (1200, 96.0), (2200, 94.0),
            (3500, 91.5), (5000, 89.5), (6500, 87.5), (8000, 86.0),
            (9000, 85.0),
        ],
        "100-50@20C": [
            (0, 100.0), (600, 98.0), (1500, 96.5), (2800, 94.5),
            (4500, 92.5), (6000, 91.0), (7500, 89.5), (9000, 88.0),
        ],
        "75-25@20C": [
            (0, 100.0), (700, 98.5), (1800, 97.0), (3000, 95.5),
            (4500, 94.0), (6000, 92.5), (7500, 91.0), (9000, 90.0),
        ],
        "75-45@20C": [
            (0, 100.0), (1000, 99.0), (2500, 98.0), (4000, 97.0),
            (5500, 96.0), (7000, 95.5), (9000, 94.5),
        ],
        "75-65@20C": [
            (0, 100.0), (1500, 99.5), (3000, 99.0), (5000, 98.0),
            (7000, 97.5), (9000, 97.0),
        ],
    }

    # ------------------------------------------------------------------
    # Compute model prediction for each case
    #
    # Each DST test cycle = 1 full rainflow cycle:
    #   DoD  = soc_start - soc_stop
    #   σ    = (soc_start + soc_stop) / 2
    #   S_T  = s_temp(T_cell_C)  [fixed at 21°C per paper Section V.B]
    #
    # fd_per_cycle = S_δ(DoD) × S_σ(σ) × S_T(T)
    #
    # This deliberately omits DST micro-cycles because those require the
    # proprietary DST signal to reproduce.  The paper's own reproduction
    # has 14% error from the same limitation.
    # ------------------------------------------------------------------
    N_MAX = 9000
    cycle_vec = np.arange(0, N_MAX + 1, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ax_data, ax_model = axes

    for (label, soc_s, soc_e), colour in zip(dst_cases, colours):
        dod     = soc_s - soc_e
        soc_m   = (soc_s + soc_e) / 2.0

        fd_per_cycle = float(fc_cycle(dod, soc_m, T_cell_C, p))
        fd_vec  = cycle_vec * fd_per_cycle
        cap_vec = xu_capacity_curve(fd_vec, p)

        # Right panel: model
        ax_model.plot(cycle_vec, cap_vec, color=colour, linewidth=1.8, label=label)

        # Left panel: digitised experimental data
        ref = fig5a_data.get(label, [])
        if ref:
            rx, ry = zip(*ref)
            ax_data.plot(rx, ry, color=colour, linewidth=1.2,
                         linestyle="--", alpha=0.6)
            ax_data.scatter(rx, ry, color=colour, s=22, zorder=4, label=label)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    for ax, title in zip(
        [ax_data, ax_model],
        ["(a) Digitised experimental data  [Fig. 5a, Xu et al. 2016]",
         "(b) Model reproduction  [this implementation, Fig. 5b equivalent]"],
    ):
        ax.set_xlim(0, N_MAX)
        ax.set_ylim(60, 107)
        ax.set_xlabel("Number of DST cycles", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.set_title(title, fontsize=11)
        ax.axhline(80, color="gray", linestyle=":", linewidth=1.0)
        ax.text(200, 80.6, "80% EoL", fontsize=8, color="gray")

    ax_data.set_ylabel("1C capacity retention (%)", fontsize=11)
    ax_model.legend(fontsize=8.5, loc="lower left",
                    title="Start–Stop SoC @ 20°C room", title_fontsize=8)

    # Annotate key match from paper: 85-25 and 100-50 have same fd (paper text, p.7)
    ax_model.annotate(
        "85-25 ≈ 100-50 (same linearised\ndegradation rate, per paper p.7)",
        xy=(9000, 84), xytext=(5500, 78),
        fontsize=7.5, color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )

    fig.suptitle(
        "Validation: Xu et al. (2016) LMO Degradation Model,  DST Cycle Test\n"
        f"T_cell = {T_cell_C}°C (paper: simulated 21°C).  "
        "Model: 1 full cycle per DST test (DoD = SoC_start − SoC_stop).\n"
        "Paper reports 14% error vs. data; residual gap here from omitted DST micro-cycles.",
        fontsize=9.5,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig



# =============================================================================
# Self-test,  run DST validation directly
# =============================================================================

if __name__ == "__main__":
    print("Running DST validation plot...")
    validate_xu_dst(save_path="xu_validation_dst.png", show=True)
    print("Done. Check xu_validation_dst.png")