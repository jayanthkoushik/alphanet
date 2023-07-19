#!/usr/bin/env python3
# %%
from pathlib import Path

import matplotlib as mpl
import pandas as pd
import seaborn as sns
from corgy import Corgy
from matplotlib.ticker import FixedFormatter, FixedLocator
from tqdm import tqdm

from alphanet._dataset import SplitLTDataset
from alphanet._plotwrap import PlotFont, PlottingConfig
from alphanet._samplers import AllFewSampler, ClassBalancedBaseSampler
from alphanet._utils import get_topk_acc
from alphanet.plot import _load_baseline_res, _load_train_res

# %%
PROFILES = {
    "paper": PlottingConfig(
        theme="light",
        context="paper",
        font=PlotFont(default="serif", math="cm"),
        bg="#ffffff",
        fg_primary="#000000",
        fg_secondary="#bbbbbb",
    ),
    "web_light": PlottingConfig(
        theme="light",
        context="notebook",
        font=PlotFont(
            default="sans-serif",
            math="stixsans",
            sans_serif=[
                "system-ui"
                " -apple-system"
                "Segoe UI"
                "Roboto"
                "Helvetica Neue"
                "Noto Sans"
                "Liberation Sans"
                "Arial"
                "sans-serif"
            ],
        ),
    ),
    "web_dark": PlottingConfig(
        theme="dark",
        context="notebook",
        font=PlotFont(
            default="sans-serif",
            math="stixsans",
            sans_serif=[
                "system-ui"
                " -apple-system"
                "Segoe UI"
                "Roboto"
                "Helvetica Neue"
                "Noto Sans"
                "Liberation Sans"
                "Arial"
                "sans-serif"
            ],
        ),
    ),
}
WIDTH_PER_CONTEXT = {"paper": 6, "notebook": 6 * 1.25}
HEIGHT_PER_CONTEXT = {"paper": 7.5, "notebook": 7.5 * 1.25}
FACET_ASPECT = 1.25


# %%
class Args(Corgy):
    n_boot: int = 10000
    rep: str = "*"
    acc_k: int = 1
    eval_batch_size: int = 1024


# %%
args = Args.parse_from_cmdline()
# args = Args(n_boot=1000, rep="1", acc_k=1, eval_batch_size=256)

# %%
df_rows = []
datasets = list(
    map(
        SplitLTDataset,
        [
            "placeslt_resnet152_crt",
            "placeslt_resnet152_lws",
            "cifarlt_resnet32_ride",
            "cifarlt_resnet34_ltr",
            "imagenetlt_resnext50_crt",
            "imagenetlt_resnext50_lws",
            "imagenetlt_resnext50_ride",
        ],
    )
)
rho_strs = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.75", "1", "1.25", "1.5", "1.75", "2"]
data_root = Path("results/main")


def _get_accs(_res):
    if args.acc_k == 1:
        _accs = {
            _split.title(): _acc for _split, _acc in _res.test_acc__per__split.items()
        }
        _accs["Overall"] = _res.test_metrics["accuracy"]
    else:
        _accs = get_topk_acc(_res, args.acc_k, args.eval_batch_size)
    return _accs


for dataset in datasets:
    baseline_res = _load_baseline_res(dataset, batch_size=1024)
    baseline_accs = _get_accs(baseline_res)

    for rho_str in rho_strs:
        for res_file in tqdm(
            list(
                (data_root / dataset / f"rho_{rho_str}").glob(
                    f"rep_{args.rep}/result.pth"
                )
            ),
            desc=f"Loading results for {dataset}|rho={rho_str}",
            unit="file",
            leave=False,
        ):
            res = _load_train_res(res_file, batch_size=args.eval_batch_size)
            assert list(res.training_config.sampler_builder.sampler_classes) == [
                AllFewSampler,
                ClassBalancedBaseSampler,
            ]

            rho = float(res.training_config.sampler_builder.sampler_args[1]["r"])
            assert float(rho_str) == rho
            assert res.train_data_info.dataset_name == dataset

            accs = _get_accs(res)

            for split in ["Many", "Medium", "Few", "Overall"]:
                df_rows.append(
                    {
                        "Dataset": dataset.proper_name,
                        "$\\rho$": rho,
                        "Split": split,
                        "Accuracy change": accs[split] - baseline_accs[split],
                    }
                )
df = pd.DataFrame(df_rows)
print(df)


# %%
def plot(profile, save=True):
    cfg = PROFILES[profile]
    cfg.config()
    mpl.rcParams["figure.constrained_layout.use"] = False

    g = sns.relplot(
        df,
        x="$\\rho$",
        y="Accuracy change",
        hue="Split",
        style="Split",
        col="Dataset",
        col_wrap=2,
        col_order=[_d.proper_name for _d in datasets],
        hue_order=["Overall", "Few", "Medium", "Many"],
        n_boot=args.n_boot,
        dashes=False,
        markers=True,
        kind="line",
        err_style="bars",
        aspect=FACET_ASPECT,
        facet_kws={"sharex": True, "sharey": True},
    )

    g.set_titles("{col_name}")
    g.set_xlabels("")
    g.set_ylabels("")

    sns.move_legend(g, loc="center", bbox_to_anchor=(0.75, 0.12), ncol=1, title="")
    g.refline(y=0, ls="-", color=cfg.palette[0], zorder=1)

    for ax in g.axes:
        ax.xaxis.set_major_locator(FixedLocator([0, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]))
        ax.xaxis.set_major_formatter(
            FixedFormatter(["$0$", "$0.5$", "", "$1$", "", "$1.5$", "", "$2$"])
        )
        ax.xaxis.set_minor_locator(FixedLocator([0.1, 0.2, 0.3, 0.4]))
    g.tick_params(axis="x", which="both", direction="in")
    g.set(
        xlim=(0, 2.05),
        ylim=(-0.25, 0.25),
        yticks=[-0.2, -0.1, 0, 0.1, 0.2],
        yticklabels=["$-0.2$", "$-0.1$", "$0$", "$0.1$", "$0.2$"],
    )
    g.despine(left=True, top=True, right=True, bottom=False, trim=True)

    _ax = g.figure.add_subplot(4, 2, 8)
    _ax.set_aspect(1 / FACET_ASPECT)  # matplotlib aspect is y/x
    for _del in [(0.9, 0), (0, 0.9)]:
        _ax.arrow(
            0.05,
            0.05,
            *_del,
            lw=mpl.rcParams["axes.linewidth"],
            length_includes_head=True,
            head_width=0.02,
            head_length=0.02,
            fc=cfg.palette[0],
            color=cfg.palette[0],
            transform=_ax.transAxes,
        )
    _ax.text(0.5, 0, "$\\rho$", transform=_ax.transAxes, va="top", ha="center")
    _ax.text(
        0,
        0.5,
        f"Top-{args.acc_k} accuracy change",
        transform=_ax.transAxes,
        va="center",
        ha="right",
        rotation="vertical",
    )
    _ax.set_xticks([])
    _ax.set_yticks([])
    sns.despine(ax=_ax, left=True, right=True, top=True, bottom=True)

    g.figure.tight_layout(h_pad=8, w_pad=3)
    width, height = WIDTH_PER_CONTEXT[cfg.context], HEIGHT_PER_CONTEXT[cfg.context]
    g.figure.set_size_inches(width, height)

    if save:
        save_root = (
            Path("paper/figures/appendix")
            if cfg.context == "paper"
            else Path("paper/figures/_www/appendix")
        )
        save_file = f"models_split_top{args.acc_k}_deltas_vs_rho"
        save_file += "_dark" if cfg.theme == "dark" else ""
        ext = ".pgf" if cfg.context == "paper" else ".svg"
        save_file += ext
        save_path = save_root / save_file
        g.figure.savefig(save_path, format=ext[1:], bbox_inches="tight")

    return g.figure


# %%
_ = plot("paper")

# %%
_ = plot("web_light")

# %%
_ = plot("web_dark")

# %%
