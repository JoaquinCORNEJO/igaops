from matplotlib import ticker as ticker
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
import pandas as pd
import os

FILEPATH = Path(os.path.realpath(__file__)).parent
FOLDER = FILEPATH / "results"
if not os.path.isdir(FOLDER):
    os.mkdir(FOLDER)

sns.set_style("whitegrid")


def set_axis(ax):
    ax.set_xlabel("Number of elements")
    ax.set_ylabel("CPU time (s)")
    ax.set_axis_on()
    ax.set(xscale="log")
    ax.set(yscale="log")
    ax.set_xlim((8, 64))
    ax.set_ylim(bottom=5e-4, top=5e1)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xticks([8, 16, 32, 64], ["8", "16", "32", "64"])


################################################################################

filename = FOLDER / "cpu_time_elliptic_fastdiag"
try:
    df = pd.read_csv(f"{filename}.csv")

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.lineplot(data=df, x="nbel", y="time", color="tab:blue", errorbar="sd", ax=ax)
    set_axis(ax)
    fig.tight_layout()
    fig.savefig(f"{filename}.pdf")
    plt.close()

except:
    testfile = "performance_fastdiag_elliptic"
    print(f"{filename} not found. Try first {testfile}")

################################################################################

filename = FOLDER / "cpu_time_elliptic_matvec"
try:
    df = pd.read_csv(f"{filename}.csv")

    for dtype in ["capacity", "conductivity"]:
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.lineplot(data=df, x="nbel", y=f"time_{dtype}", hue="degree", ax=ax)
        set_axis(ax)
        fig.tight_layout()
        fig.savefig(f"{filename}_{dtype}.pdf")
        plt.close()

except:
    testfile = "performance_matvec_elliptic"
    print(f"{filename} not found. Try first {testfile}")

################################################################################

filename = FOLDER / "cpu_time_sptm_fastdiag"
try:
    df = pd.read_csv(f"{filename}.csv")

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.lineplot(data=df, x="nbel", y="time", color="tab:blue", errorbar="sd", ax=ax)
    set_axis(ax)
    fig.tight_layout()
    fig.savefig(f"{filename}.pdf")
    plt.close()

except:
    testfile = "performance_fastidiag_sptm"
    print(f"{filename} not found. Try first {testfile}")

################################################################################

filename = FOLDER / "cpu_time_sptm_matvec"
try:
    df = pd.read_csv(f"{filename}.csv")

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.lineplot(data=df, x="nbel", y="time", hue="degree", ax=ax)
    set_axis(ax)
    fig.tight_layout()
    fig.savefig(f"{filename}.pdf")
    plt.close()

except:
    testfile = "performance_matvec_elliptic"
    print(f"{filename} not found. Try first {testfile}")

################################################################################
# COMPARISONS (combination of former figures)
################################################################################


def plot_performance_envelope(
    df: pd.DataFrame, nbel: str, time: str, deg: str, ax, color="teal", label=None
):
    subgroup = df[df[deg] > 1]
    envelope = subgroup.groupby(nbel)[time].agg(["min", "max"]).reset_index()
    ax.fill_between(
        envelope[nbel],
        envelope["min"],
        envelope["max"],
        color=color,
        alpha=0.5,
        label=label,
    )
    return


try:
    fig, ax = plt.subplots(figsize=(5, 5))

    #
    filename = FOLDER / "cpu_time_elliptic_matvec"

    df = pd.read_csv(f"{filename}.csv")
    plot_performance_envelope(
        df,
        "nbel",
        "time_conductivity",
        "degree",
        ax,
        color="tab:orange",
        label="MF-WQ w. conductivity matrix",
    )
    plot_performance_envelope(
        df,
        "nbel",
        "time_capacity",
        "degree",
        ax,
        color="tab:green",
        label="MF-WQ w. capacity matrix",
    )

    #
    filename = FOLDER / "cpu_time_elliptic_fastdiag"

    df = pd.read_csv(f"{filename}.csv")
    plot_performance_envelope(
        df,
        "nbel",
        "time",
        "degree",
        ax,
        color="tab:blue",
        label="Apply preconditioner",
    )

    ax.set_xlabel("Number of elements")
    ax.set_ylabel("CPU time (s)")
    ax.set(xscale="log")
    ax.set(yscale="log")
    ax.set_xlim((8, 64))
    ax.set_ylim(bottom=1e-4, top=1e2)
    x_ticks = [8, 16, 32, 64]
    ax.set_xticks(x_ticks, [rf"${str(x)}^3$" for x in x_ticks])
    ax.legend(title=rf"Degree $p=2,\ldots,20$", loc="upper left", fontsize=11)
    ax.minorticks_off()
    fig.tight_layout()
    fig.savefig(FOLDER / "comparison_elliptic.pdf")
    plt.close()

except:
    print("Could not perform comparison in elliptic problems.")


try:
    fig, ax = plt.subplots(figsize=(5, 5))

    #
    filename = FOLDER / "cpu_time_sptm_matvec"

    df = pd.read_csv(f"{filename}.csv")
    plot_performance_envelope(
        df,
        "nbel",
        "time",
        "degree",
        ax,
        color="tab:orange",
        label="MF-WQ w. Picard matrix",
    )

    #
    filename = FOLDER / "cpu_time_sptm_fastdiag"

    df = pd.read_csv(f"{filename}.csv")
    plot_performance_envelope(
        df,
        "nbel",
        "time",
        "degree",
        ax,
        color="tab:blue",
        label="Apply preconditioner",
    )

    ax.set_xlabel("Number of elements")
    ax.set_ylabel("CPU time (s)")
    ax.set(xscale="log")
    ax.set(yscale="log")
    ax.set_xlim((8, 64))
    ax.set_ylim(bottom=1e-4, top=1e2)
    x_ticks = [8, 16, 32, 64]
    ax.set_xticks(x_ticks, [rf"${str(x)}^3$" for x in x_ticks])
    ax.legend(title=rf"Degree $p=2,\ldots,20$", loc="upper left", fontsize=11)
    ax.minorticks_off()
    fig.tight_layout()
    fig.savefig(FOLDER / "comparison_sptm.pdf")
    plt.close()

except:
    print("Could not perform comparison in space-time problems.")
