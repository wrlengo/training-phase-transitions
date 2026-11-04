# Computo pre-submission checklist

Status of the manuscript package as of 2026-11-04. Items marked OPEN block
submission; items marked OPTIONAL can be answered in the response letter.

## Blocking

- [ ] **Affiliation, city and country** in `_quarto.yml` are `TBD`.
- [ ] **ORCID** in `_quarto.yml` is `TBD`.
- [ ] **Zenodo DOI**: deposit the archive (per-step metrics, configs, figure
      code, checkpoints) and insert the DOI in the Code availability statement.
      The text currently says "will be deposited at Zenodo".
- [ ] **Public repository**: `github.com/wrlengo/training-phase-transitions`
      does not exist yet. SSH authentication now works (`ssh -T git@github.com`
      returns the `wrlengo` greeting, key id 168775659), so the remaining step is
      to create the repository, add a `.gitignore` for checkpoints, logs and
      caches, and push.
- [x] **Local render**: Quarto 1.11.9 and the Computo extension are installed
      (conda env `computo-render`). Both targets render cleanly into
      `_output/main.html` and `_output/main.pdf`: 11 figures embedded, five
      bold-text tables, no unresolved cross-references, no LaTeX errors.
- [ ] **CI + demo**: add a workflow that runs the two demo commands in the
      README. Computo reviewers may run the workflow; without it the demo claim
      is untested.
- [ ] **Read the reviewer-facing claims** once more: the architecture claim must
      stay scoped to the protocol-matched Transformer/MLP pair. Do not strengthen
      it into "the ConvNet comparison proves architecture dependence".

## Manuscript consistency (checked in this pass)

- [x] Abstract is about 200 words, contains one measured number (`0.035`), and
      does not mention the log-alignment ratio.
- [x] The six contributions are word-for-word identical in Section 1 and
      Section 9.
- [x] `beta` and finite-size scaling are labelled exploratory and appear only in
      the designated sections.
- [x] The only use of "phase transition" is guarded by the three-signature rule;
      the Transformer result is written as "the order parameter does not track",
      never as "no phase transition".
- [x] `S` is defined as `S / ln K` wherever the order parameter is introduced.
- [x] `eta` is used only for the learning rate and `T_attn` only for the model
      temperature; this is stated explicitly in Section 2.1.
- [x] Every experiment directory referenced in the text exists.
- [x] All 26 bibliography entries resolve; no entry was added without a
      Crossref/arXiv record.

## Optional robustness (not run; disclosed as limitations)

- [ ] Bin-count sensitivity of `m` (rerun the ConvNet endpoints with 50 and 200
      bins on the saved checkpoints). If this reverses the architecture
      comparison, the conclusion changes; it is the first thing a referee may ask.
- [ ] Dense `eta` grid in `[0.030, 0.040]` with five seeds, to identify `beta`.
- [ ] Cross experiments: ConvNet on modular addition, Transformer on CIFAR-10.
      These are the decisive tests for the task/architecture confound.
- [ ] Direct measurement of `Q_tilde` under the AdamW metric, to test the
      proposed correction in Section 6.3.

## Submission

- [ ] Format check against the Computo author guidelines
      (`https://computo-journal.org/site/guidelines-authors.html`).
- [ ] Upload `main.qmd`, `_quarto.yml`, `references.bib`, `environment.yml`,
      `code/`, `data/`, `figures/` and the rendered PDF.
- [ ] OpenReview submission form: title, abstract, keywords, authors, licence,
      code link, data link.
- [ ] Keep the generative-AI disclosure as a separate section before the
      references, as required by the journal.
