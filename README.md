# Which observables track training phase transitions?

Code and data for *"Which Observables Track Training Phase Transitions? A
Systematic Comparison Across ConvNets, Transformers, and MLPs"* (Rui Wang, Yi Ma).

One definition of the thermodynamic observables, one scanning protocol and one
fitting routine are applied to three architectures: a ConvNet on CIFAR-10, and a
Transformer and an MLP on modular arithmetic. The repository contains the
per-step metrics of every run, the configuration of every run, the 15 final
checkpoints, and the scripts that turn those records into the tables and figures
of the manuscript.

Archived releases: `v1.0.1`&ndash;`v1.0.3`, concept DOI [10.5281/zenodo.23957416](https://doi.org/10.5281/zenodo.23957416) (resolves to the latest version).

## Layout

```
├── phase*/                  per-run outputs: metrics_*.csv, config.json, checkpoints
├── *.py                     training and analysis scripts (one per experiment)
├── paper/
│   ├── main.qmd             manuscript source
│   ├── data/                compact analysis-ready CSV/JSON
│   ├── figures/             vector PDFs and PNG previews
│   ├── code/                make_data.py, figures.py, training/ (14 scripts)
│   ├── _extensions/computo  vendored Computo Quarto extension
│   └── environment.yml      conda environment for the paper
├── TRAINING_CONFIGS.md      executed settings of every run
└── training_configs_summary.csv
```

`data/` (the CIFAR-10 download, ~340 MB) is excluded; torchvision fetches it on
first use.

## Reproducing the analysis

```bash
conda env create -f paper/environment.yml
conda activate computo-training-transitions
python paper/code/make_data.py     # phase*/ -> paper/data/
python paper/code/figures.py       # paper/data/ -> paper/figures/
cd paper && quarto render main.qmd
```

The manuscript is rendered with the Computo Quarto extension, vendored in
`paper/_extensions/computo`; the GitHub Actions workflow in
`.github/workflows/publish.yml` renders it and deploys the HTML to GitHub Pages on
every push to `main`.

## Demonstration run

The training scripts accept reduced settings:

```bash
python paper/code/training/phase3_free_energy.py --lrs 1e-3 1e-2 1e-1 --seeds 0 \
    --total_steps 5000 --log_every 1000 --out_dir demo/convnet
python paper/code/training/phase7_modular_fe_scan.py --arch transformer \
    --lrs 1e-4 1e-3 1e-2 --seeds 0 --total_steps 5000 --out_dir demo/transformer
```

These validate the workflow, not the conclusions; the paper uses the full grids
listed in Table 1 of the manuscript.

## Licence

Text and figures: CC-BY 4.0. Code: MIT.
