# alphanet
This repository contains code for the AlphaNet project. This file
documents setup and usage for running experiments. The instructions
below assume that you are using a Unix-like system (e.g., Linux or
macOS). The code should also work on Windows, but this has not been
tested, and the commands will need to be modified.

## Setup
Download the repository, or clone it using `git`. Running the code requires
**Python 3.8 or higher**. Dependencies can be installed using
[Poetry](https://python-poetry.org/), preferably in a virtual environment:

<!-- cspell: disable -->
```bash
$ python -m venv .venv  # create a virtual environment named '.venv'
$ source .venv/bin/activate  # activate the virtual environment
$ poetry install --only main  # install dependencies
```
<!-- cspell: enable -->

## Data setup

**Pre-formatted data for all datasets used in the paper are available
for download from
[Google Drive](https://drive.google.com/file/d/1RfbzNovAVOQJTlHNfUgCbl34tOGLxNv9/view?usp=share_link).
Extract the `tar` file to the root of the repository (i.e., you should
have `data/ImageNetLT` at the root of the repository), then symlink
`config/datasets.toml` to `config/datasets.toml.template` (e.g., using
`ln -s config/datasets.toml.template config/datasets.toml`). The below
instructions are only required if you want to prepare your own datasets.
Otherwise you can skip to the next section.**

The code expects data in a particular format. Datasets should be
prepared into this format and linked using the config file
`config/datasets.toml`. `config/datasets.toml.template` shows an example
of this file. The first item in the file specifies a dataset named
`imagenetlt_resnext50_crt` which use cRT features from ImageNet-LT. The
expected file structure for this dataset is as follows:

```bash
data
├── ImageNetLT/
│   ├── label_names.txt
│   ├── splits/
│   │   ├── few.txt
│   │   ├── medium.txt
│   │   ├── many.txt
│   ├── features/
│   │   ├── resnext50_crt/
│   │   │   ├── train.pkl
│   │   │   ├── val.pkl
│   │   │   ├── test.pkl
│   ├── classifiers/
│   │   ├── resnext50_crt.pth
```

`few.txt`, `medium.txt`, and `many.txt` should contain the label numbers
(one per line) for the few, medium, and many splits respectively.

The feature files `train.pkl`, `val.pkl`, and `test.pkl` should contain
`dict`s (corresponding to train, val, and test data) with keys
`'feats'`, `'labels'`, and `'idxs'`. `feats` should be a `torch.Tensor`
of shape `n x d` (`samples x features`). `labels` should a `list` of
length `n`, and `idxs` should be the a `torch.Tensor` of length `n` with
sample indices, i.e., `labels[i]` should be the label for
`feats[idxs[i]]`.

The classifier file (`resnext50_crt.pth` in the above example) can use
different formats based on the classifier loader. For the simple case of
a `LinearClassifierLoader` which is used in the example file, the
classifier file should contain a `dict` with keys `'weight'` and
`'bias'`. `weight` should be a `torch.Tensor` with shape `c x d`
(`classes x features`), and `bias` should be a `torch.Tensor` with shape
`c`.

The optional `label_names.txt` file should contain the label names (one
per line) for the dataset, ordered by label number, `0` to `c - 1`.

<!-- cSpell:ignore run_printres, run_showcombos, run_genplots, run_makeplot -->

## Running experiments
**NOTE: The code contains many debug statements, which can significantly
slow down training. To disable these, pass `-O` to the Python
interpreter (e.g., `python -O run_train.py ...`)**.

The main script for running experiments is `run_train.py`. All training
parameters are passed as command line arguments. These are documented
below:

* `--dataset`: Name of a dataset defined in `config/datasets.toml`
    (e.g., `imagenetlt_resnext50_crt` based on `datasets.toml.template`).
    **This is the only required argument**.

* `--save-file`: Optional file to save training results to. If not
    specified, results are not saved.

* `--ckpt-dir`: Directory to save/load checkpoints to/from. If not
    specified, checkpoints are not used.

* `--n-ckpt`: Number of most recent checkpoints to keep (default: 2).

* `--no-load-from-ckpt`: By default, if `--ckpt-dir` is specified, the
    latest checkpoint is loaded before training. This flag disables this
    behavior.

**AlphaNet parameters:**

* `--alphanet:hdims`: Hidden dimension sizes for AlphaNet, as a space
    separated list of numbers (default: `32 32 32`, i.e., a three-layer
    network with 32 hidden units per layer).

* `--alphanet:hact`: Name of an activation function in
    `torch.nn.functional` (default: `leaky_relu`).

**Training parameters:**

* `--training:n-neighbors`: Number of nearest neighbors (excluding self)
    to use (default: 5).

* `--training:nn-dist`: Distance metric for nearest neighbors. Can be
    one of `cosine`, `euclidean`, or `random` (default: `euclidean`).
    `random` is provided as a baseline, and will generate random distances
    for every pair of points.

* `--training:ptopt:optim-cls`: Name of an optimizer class,
    specifically, any sub-class of `torch.optim.Optimizer` (default:
    `AdamW`).

* `--training:ptopt:optim-params`: Parameters for the optimizer, as a
    comma separated string of `key=value` pairs (default: `"lr=0.001"`).

* `--training:ptopt:lr-sched-cls`: Name of a learning rate scheduler
    class, specifically, any sub-class of
    `torch.optim.lr_scheduler._LRScheduler` (default: `StepLR`).

* `--training:ptopt:lr-sched-params`: Parameters for the learning rate
    scheduler, as a comma separated string of `key=value` pairs (default:
    `"step_size=10,gamma=0.1"`).

* `--training:train-epochs`: Number of epochs to train (default: 25).

* `--training:min-epochs`: Minimum number of epochs to use when
    selecting the best model (default: 5, i.e., the model with the best
    validation accuracy after 5 epochs is selected).

* `--training:train-batch-size`: Batch size to use while training
    (default: 64).

* `--training:eval-batch-size`: Batch size to use while evaluating
    (default: 1024).

* `--training:sampler-builder:sampler-classes`: One or more classes to
    use for sampling training data. These should be one of the public
    classes in `alphanet/_samplers.py`. Check the doctrings for these
    classes for more information
    (default: `AllFewSampler ClassBalancedBaseSampler`).

* `--training:sampler-builder:sampler-args`: Arguments for the sampler
    classes, as a comma separated string of `key=value` pairs, one per
    sampler class. Check the doctrings for the sampler classes for more
    information (default: `"" "r=0.1"`).

* `--training:tb-logs`: Tensorboard log directory. If not specified,
    tensorboard logs are not written.

Example command:

```bash
python -O run_train.py \
    --dataset imagenetlt_resnext50_crt \
    --save-file results/imagenetlt_resnext50_crt/run_1/result.pkl \
    --ckpt-dir results/imagenetlt_resnext50_crt/run_1/ckpts \
    --training:tb-logs results/imagenetlt_resnext50_crt/run_1/tb \
    --training:sampler-builder:sampler-args "" "r=0.25"
```

This trains AlphaNet using default parameters, and only changes the
sampling ratio for the `ClassBalancedBaseSampler` to 0.25. Results and
logs are saved to `results/imagenetlt_resnext50_crt/run_1`.

### Generating baseline results
AlphaNet can be run in dummy mode as a sanity check to obtain baseline
results. This can be done using the `run_baseline.py` script. This
script takes the dataset name (as defined in `config/datasets.toml`) as
a command line argument (`--dataset`), and saves the baseline results to
the path specified in the config file under `baseline_eval_file`.
Example:

```bash
python -O run_baseline.py --dataset imagenetlt_resnext50_crt
```

## Printing results
To summarize generated results from multiple experiments, the
`run_printres.py` script can be used. For demonstration of the usage,
let us assume a results directory structure as follows:

```bash
results/
├── imagenetlt_resnext50_crt/
│   ├── rho_0.1/
│   │   ├── run_1/
│   │   │   ├── result.pkl
│   │   ├── run_2/
│   │   │   ├── result.pkl
│   ├── rho_0.2/
│   │   ├── run_1/
│   │   │   ├── result.pkl
│   │   ├── run_2/
│   │   │   ├── result.pkl
│   ├── rho_0.3/
│   │   ├── run_1/
│   │   │   ├── result.pkl
│   │   ├── run_2/
│   │   │   ├── result.pkl
```

The arguments to `run_printres.py` are as follows:

* `--base-res-dir`: Base directory with result files
    (`results/imagenetlt_resnext50_crt` in the above example).

* `--rel-exp-paths`: Relative paths to individual experiment folders
    (`rho_0.1 rho_0.2 rho_0.3` in the above example).

* `--exp-names`: Names for the individual experiments. This argument is
    optional, and only changes the display names in the printed results.
    If not specified, the experiment folder names are used.

* `--res-files-pattern`: Regex pattern for result files
    (`"run_*/result.pkl"` for the above example).

* `--exp-prefix` and `--exp-suffix`: Prefix/suffix to add to the
    experiment names in the printed results. These arguments are
    optional, and only change the display names in the printed results.
    If not specified, the experiment names are not prefixed/suffixed.

So, the command to print results for the above example would be:

<!-- cSpell: disable -->
```bash
$ python run_printres.py --base-res-dir results/imagenetlt_resnext50_crt --rel-exp-paths rho_0.1 rho_0.2 rho_0.3 --exp-names 0.1 0.2 0.3 --res-files-pattern "run_*/result.pkl" --exp-prefix "AlphaNet (r=" --exp-suffix ")"
Experiment AlphaNet (r=0.1): 100%|████████████| 10/10 [00:00<00:00, 20.88file/s]
Experiment AlphaNet (r=0.2): 100%|████████████| 10/10 [00:00<00:00, 20.92file/s]
Experiment AlphaNet (r=0.3): 100%|████████████| 10/10 [00:00<00:00, 20.46file/s]
Loading: 100%|████████████████████████████| 3/3 [00:01<00:00,  2.07experiment/s]
Loading baselines: 100%|█████████████████████| 1/1 [00:00<00:00, 22.51dataset/s]


ImageNet-LT | cRT:
==================

Experiment                Few         Med.         Many      Overall
----------------  -----------  -----------  -----------  -----------
Baseline                 27.4         46.2         61.8         49.6
AlphaNet (r=0.1)  47.6^±1.60^  37.1^±0.76^  53.8^±0.62^  45.0^±0.41^
AlphaNet (r=0.2)  45.7^±1.56^  39.0^±0.78^  55.7^±0.79^  46.3^±0.52^
AlphaNet (r=0.3)  42.9^±1.29^  40.4^±0.72^  57.0^±0.68^  47.1^±0.47^

Experiment,Few,Medium,Many,Overall
Baseline,27.4,46.2,61.8,49.6
AlphaNet (r=0.1),47.6,37.1,53.8,45.0
AlphaNet (r=0.2),45.7,39.0,55.7,46.3
AlphaNet (r=0.3),42.9,40.4,57.0,47.1
```
<!-- cSpell: enable -->

### Showing learned combinations
AlphaNet updates few split classifiers using a linear combination of
base split classifiers. The learned linear combinations can be shown
using the `run_showcombos.py` script. The script can show aggregate
results from multiple runs of an experiment. The arguments to the
script are as follows:

* `--base-res-dir`: Directory with result files for an experiment.

* `--res-files-pattern`: Regex pattern for result files (default:
    `"**/*.pth"`).

* `--select-by`: `'acc'` or `'acc_delta'`. This argument specifies the
    way classifiers are selected for showing the learned combinations.
    `acc` selects the classifiers based on the final test accuracy,
    while `acc_delta` selects the classifiers based on the accuracy
    change, compared to the baseline (default: `'acc'`).

* `--no-select-max`: By default, classifiers are selected based on
    the maximum accuracy or accuracy change. This argument will make
    the script instead use the minimum.

* `--select-n`: Number of classifiers to show (default: 5).

* `--mistake-n`: The script shows the top misclassifications for each
    selected classifier. This argument specifies the number of
    misclassifications to show. If not specified, all mistakes are
    shown.

* `--eval-batch-size`: Batch size for evaluation (default: 1024).

A sample output from the script is as follows:

<!-- cSpell: disable -->
```bash
$ python run_showcombos.py --base-res-dir results/imagenetlt_resnext50_crt/rho_0.1 --res-files-pattern "run_1/result.pkl" --select-n 1 --mistake-n 3
[nltk_data] Downloading package wordnet to .nltk...
[nltk_data]   Package wordnet is already up-to-date!
Loading:   0%|                                                                            | 0/1 [00:00<?, ?file/s]INFO:03:06:40:stacking 'test' features...
INFO:03:06:40:stacking 'test' features...done
INFO:03:06:40:Engine run starting with max_epochs=1.
                                                                                                                 INFO:03:06:42:Epoch[1] Complete. Time taken: 00:00:01.736
INFO:03:06:42:Engine run complete. Time taken: 00:00:01.747
Loading: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.98s/file]
INFO:03:06:42:Engine run starting with max_epochs=1.
INFO:03:06:43:Epoch[1] Complete. Time taken: 00:00:00.769
INFO:03:06:43:Engine run complete. Time taken: 00:00:00.779
********************************************
[Eskimo dog] = [Eskimo dog]
               + (-0.18)[Siberian husky]
               + (-0.066)[malamute]
               + (0.22)[Norwegian elkhound]
               + (0.26)[timber wolf]
               + (0.27)[Cardigan]

Training samples: 17
Baseline accuracy: 0.20
Mean AlphaNet accuracy: 0.76

Top baseline misclassifications (total 40 mistakes / 50 predictions),
and corresponding mean AlphaNet misclassifications
(total 12 mistakes / 50 predictions, across 1 repetition(s)):
        malamute ('medium' split): 16 -> 2.00 (2/1)
                |-> as 'Eskimo dog': 3 -> 34.00 (34/1)
        Siberian husky ('many' split): 10 -> 1.00 (1/1)
                |-> as 'Eskimo dog': 5 -> 41.00 (41/1)
        white wolf ('many' split): 4 -> 3.00 (3/1)
                |-> as 'Eskimo dog': 0 -> 3.00 (3/1)

Top AlphaNet misclassifications:
        white wolf ('many' split): 4 -> 3.00 (3/1)
                |-> as 'Eskimo dog': 0 -> 3.00 (3/1)
        collie ('few' split): 1 -> 2.00 (2/1)
                |-> as 'Eskimo dog': 0 -> 1.00 (1/1)
        malamute ('medium' split): 16 -> 2.00 (2/1)
                |-> as 'Eskimo dog': 3 -> 34.00 (34/1)
********************************************
```
<!-- cSpell: enable -->

## Generating plots
`run_genplots.sh` generates all plots used in the paper and the website.
To generate individual plots, use the `run_makeplot.py` script. Refer to
the script and command line help for more details.
