# Which observables track training phase transitions?

Submission package for Computo. The manuscript compares one order parameter,
the normalised weight-histogram entropy, across a ConvNet, a Transformer and
an MLP, and reports where it does and does not track a learning-rate-driven
training transition.

## Layout

```
paper/
├── main.qmd                 manuscript source (sections, appendix, code chunks)
├── _quarto.yml              Computo metadata, abstract, formats
├── references.bib           26 entries, all checked against Crossref/arXiv
├── environment.yml          conda environment (Python 3.11, PyTorch 2.3)
├── COMPUTO_FRAMEWORK.md     working notes: claim-evidence ledger and section plan
├── code/
│   ├── make_data.py         raw phase*/ outputs -> compact CSV in data/
│   ├── figures.py           data/ -> figures/*.pdf and *.png (11 figures)
│   └── training/            the 14 training and analysis scripts used for the runs
├── data/                    analysis-ready CSV/JSON; every quoted number comes from here
└── figures/                 vector PDFs and PNG previews
```

Raw outputs (`metrics_*.csv`, `config.json`, checkpoints) live one level up in
`../phase*/` and are never modified by the paper code.

## Quick start

```bash
conda env create -f environment.yml
conda activate computo-training-transitions
python code/make_data.py          # rebuilds data/ from ../phase*/
python code/figures.py            # rebuilds figures/
quarto add computorg/computo-quarto-extension
quarto render                     # renders main.qmd with the Computo format
```

Every number quoted in the prose comes from `data/key_numbers.json` or from the
CSVs in `data/`, and the tables and figures are generated from those files at
render time. If `make_data.py` and `figures.py` have not been rerun, the
manuscript will show stale values; the render will still succeed, so check the
git status of `data/` before submission.

## Demonstration run

The training scripts accept reduced settings, so the pipeline can be exercised
without reproducing a result:

```bash
python code/training/phase3_free_energy.py --lrs 1e-3 1e-2 1e-1 --seeds 0 \
    --total_steps 5000 --log_every 1000 --out_dir demo/convnet
python code/training/phase7_modular_fe_scan.py --arch transformer \
    --lrs 1e-4 1e-3 1e-2 --seeds 0 --total_steps 5000 --out_dir demo/transformer
```

These take minutes. They validate the workflow, not the conclusions; the paper
uses the full grids listed in Table 1.

## Full configuration

The executed settings of every run are recorded in `../phase*/config.json` and
summarised in `../TRAINING_CONFIGS.md`. The ConvNet was specified at 1.6M steps
and executed for 200k; both values are in the config files, and the paper
quotes the executed budget.

## Repository and continuous integration

The public repository is `github.com/wrlengo/training-phase-transitions`, release
`v1.0.0`. The workflow in `.github/workflows/publish.yml` renders this manuscript
with the vendored Computo extension on every push to `main` and deploys the HTML
to `wrlengo.github.io/training-phase-transitions`. The archived release carries
the Zenodo DOI.

## What is not in this package

- The CIFAR-10 download (~340 MB); torchvision fetches it on first use.
- The two optional robustness experiments discussed in the limitations: bin
  sensitivity for the entropy estimator, and a denser learning-rate grid in
  `[0.030, 0.040]` for the exponent.

## Licences

Text and figures: CC-BY 4.0. Code: MIT.
