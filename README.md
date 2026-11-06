# Recycled Corpora, Divergent Labels

Code and outputs for **"Recycled Corpora, Divergent Labels: Re-examining Cross-Corpus Generalisation in Roman Urdu Sentiment Classification"** (Jawairia Iftikhar). The paper is in preparation.

The study audits four public Roman Urdu sentiment corpora for shared sentences and label agreement. It builds a decontaminated cross-corpus benchmark, and on that benchmark it compares word and character TF-IDF classifiers with mBERT and XLM-R. Every model is trained on one corpus and tested on each of the others.

## Main findings

- The corpora are not independent. Aimlab contains 5,245 UCI sentences with the same labels. Khubaib01 contains 15,433 UCI and 15,884 Aimlab sentences, and gives 16% and 27% of them a different label (Cohen's κ 0.755 and 0.591).
- After decontamination, macro-F1 falls significantly in 23 of 24 three-class transfer directions, by up to 29.5 points.
- XLM-R does not transfer better than a character n-gram baseline. It is better only when Khubaib01 is the test corpus.

## Repository contents

```
notebooks/
  01_data_baselines_transformers.ipynb   # loading, cleaning, decontamination, splits,
                                         # TF-IDF baselines, mBERT and XLM-R fine-tuning,
                                         # generalisation drops with bootstrap CIs
  02_analysis_and_figures.ipynb          # label agreement, model-vs-both-label-sets analysis,
                                         # paired bootstrap tests (Holm), Figures 1-4
outputs/
  overlaps/   exact_overlaps.csv, near_duplicate_pairs.csv
  splits/     train/validation/test assignments for each corpus
  results/    per-run scores, generalisation drops, corpus summary
  analysis/   label agreement, crosstabs, paired tests, macro-F1 with CIs
  figures/    fig1-fig4 (PNG, 300 dpi)
```

## Data

The corpora are **not** redistributed here. Download them from their original sources and upload the files to Colab:

| Corpus | Source | Licence |
|---|---|---|
| UCI | Roman Urdu Data Set, UCI Machine Learning Repository (Sharf and Rahman, 2018) | [confirm] |
| Aimlab | `Aimlab/Sentiment-Analysis-Roman-Urdu` on Hugging Face | none stated |
| RUWV-NSR | Mendeley Data, doi:10.17632/v5jfhsvtmd.5 (Soomro et al., 2024) | CC BY 4.0 |
| Khubaib01 | `Khubaib01/RomanUrdu-NLP-Sentiment-Corpus` on Hugging Face, doi:10.57967/hf/7931 | Apache-2.0 |

Notebook 1 recognises each file from its column names, so file names do not matter. If the Aimlab or Khubaib01 file is not uploaded, it loads that corpus from Hugging Face.

Released split and prediction files contain sentence text only for RUWV-NSR and Khubaib01, whose licences allow redistribution. Aimlab and UCI rows are given as row numbers and labels, so you need the original files to rebuild them.

## How to reproduce

1. Open `01_data_baselines_transformers.ipynb` in Google Colab and select a **T4 GPU** runtime.
2. Upload the four corpus files, then run all cells. The transformer stage runs 42 fine-tuning jobs (2 models × 7 source corpora across both tracks × 3 seeds) and takes about 3 hours on a T4. Each finished run is saved, so you can stop and resume. The notebook downloads `roman_urdu_paper_outputs.zip` at the end.
3. Open `02_analysis_and_figures.ipynb` on a CPU runtime, upload that zip, and run all cells.

Fixed settings: split seed 42; training seeds 13, 42 and 2026; training size capped at 15,500 (three-class) and 8,716 (binary); XLM-R base and bert-base-multilingual-cased, learning rate 2e-5, batch 32, 3 epochs, maximum length 128.

Expected corpus sizes after cleaning and decontamination: UCI 19,376; Aimlab 11,353; RUWV-NSR 20,400; Khubaib01 70,581. RUWV-NSR may come out as 20,401 because of near-neighbour tie-breaking. The reported results use 20,400.

## Requirements

Python 3.10+, pandas, numpy, scikit-learn, pyarrow, matplotlib, torch, transformers. Colab provides all of these.

## Citation

If you use this code, please cite the repository until the paper is available:

```
Iftikhar, J. (2026) Recycled Corpora, Divergent Labels: code and data for cross-corpus Roman Urdu sentiment classification. GitHub. Available at: <repository URL>.
```

## Licence

Code: MIT. The corpora keep their original licences (see the table above).

## Acknowledgement of AI assistance

An AI assistant (Claude, Anthropic) helped write the analysis code and draft the paper. The author ran all experiments and checked all reported outputs.
