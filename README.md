# Vega: eligibility-aware patient-to-trial search

CSD358 IR mid-term hackathon, **Track T6: vertical search (medicine and science)**.

Vega is named after one of the brightest stars in the night sky. Paste a patient's case description and it ranks the clinical trials on ClinicalTrials.gov that the patient could join. The goal is to rank trials the patient is **eligible** for above trials they would be **excluded** from. To do that, Vega uses the structure of a trial record (title, summary, conditions, interventions, inclusion criteria, exclusion criteria, and age and sex limits), not just its flat text.

It works in two stages:

1. **Lexical stage.** An inverted index built from scratch scores each trial with BM25F over seven zones, then applies the age/sex eligibility test, pseudo-relevance feedback from the trials' *conditions* zone, and an exclusion penalty.
2. **Multi-view neural re-ranking.** A biomedical cross-encoder (MedCPT), used zero-shot with no fine-tuning, reads the patient note against three *separate* views of each of the top 100 trials: its topic, its inclusion criteria and its exclusion criteria. Agreement with the inclusion criteria is rewarded and agreement with the exclusion criteria is penalised, then fused with the structured eligibility test.

Author: K S Nithurshen (2410110157), ks622@snu.edu.in

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13 on macOS (14 cores, 48 GB RAM, Apple GPU). The build needs about 9 GB of RAM and about 5 GB of disk.

The first time neural re-ranking runs, the MedCPT model (about 440 MB, public domain) is downloaded from HuggingFace. It runs on Apple GPUs (MPS), on CUDA, or more slowly on a CPU.

## Reproduce everything

```bash
python -m vega.fetch              # topics, qrels and the 1.8 GB trial corpus (polite, robots.txt-checked)
python -m vega.build              # parse 375,580 trials and build the inverted index (about 2 min)
python -m vega.evaluate --stem    # lexical stage: tune on 2021, test on 2022, ablations (about 45 min)
python -m vega.evaluate neural    # multi-view MedCPT re-ranking: tune 4 weights on 2021, test on 2022 (about 15 min)
python -m vega.evaluate figures   # redraw figures and LaTeX tables from results/summary.json
python -m vega.trials stats       # corpus-wide criteria statistics used in the report
python -m vega.report             # fill report/main.tex from results/summary.json
streamlit run app.py              # the UI
```

## Data

**TREC Clinical Trials track 2021 and 2022** (NIST). This is a standard benchmark with expert relevance judgements:

- **Corpus:** a snapshot of ClinicalTrials.gov from 27 April 2021 with 375,580 trials, from [trec-cds.org](https://www.trec-cds.org/2021.html).
- **Topics:** [2021](https://trec.nist.gov/data/trials/topics2021.xml) (75 patients) and [2022](https://trec.nist.gov/data/trials/topics2022.xml) (50 patients). Each is a 5–10 sentence patient description written by clinicians; they are not real patients.
- **Judgements:** [qrels2021](https://trec.nist.gov/data/trials/qrels2021.txt) and [qrels2022](https://trec.nist.gov/data/trials/qrels2022.txt). Each judged trial is 2 = eligible, 1 = excluded or 0 = not relevant.

We tune every setting on 2021 and report on 2022. The official main metric is nDCG@10 with gains 2/1/0; P@10 and RR count only eligible trials as relevant.

## Code map

| File | IR role |
|---|---|
| `vega/fetch.py` | Polite downloader: robots.txt check, delay between requests, resumable |
| `vega/trials.py` | Parses the XML into 7 zones and age/sex parameters. Criterion polarity normalisation moves criteria that start with "No…" into the exclusion zone |
| `vega/text.py` | Tokenisation, folding, stop words, Porter stemmer (written from scratch) |
| `vega/patient.py` | Patient-note processing: age/sex extraction, clinical abbreviation expansion, NegEx-style negation, family-history (experiencer) detection |
| `vega/index.py` | Parallel inverted-index build: per-zone tf, positional postings, champion lists, parametric age/sex arrays, skip-pointer intersection, phrase queries |
| `vega/boolean.py` | Boolean query parser: AND/OR/NOT, phrases, zone fields, `age:` and `sex:` filters |
| `vega/rank.py` | TF-IDF, BM25, BM25F accumulators; index elimination; champion lists; heap and partial-sort top-K; age/sex eligibility; conditions-zone pseudo-relevance feedback; polarity-aware penalties; two-stage re-ranking |
| `vega/neural.py` | MedCPT cross-encoder and the three trial views (topic, inclusion, exclusion), plus score fusion |
| `vega/evaluate.py` | nDCG@10, P@10, RR, R@1000, Excluded@10, judged@10 and condensed-list metrics; tuning on 2021; testing on 2022; ablations; significance tests |
| `vega/report.py` | Fills the LaTeX report from the results |
| `app.py` | Streamlit UI: patient matching with explanations, Boolean search, index inspector, evaluation |

The earlier Polaris prototype (citation recommendation over OpenAlex) is kept in `archive/polaris/` for reference.

## What works

- End-to-end download, parsing, indexing, ranking and UI over all 375,580 trials, at a few hundred milliseconds per patient.
- Full evaluation on the official TREC judgements, with a 2021 tuning / 2022 test split, ablations, efficiency measurements and paired significance tests.
- Test results (TREC CT 2022, 50 patients; everything tuned on 2021):

  | System | nDCG@10 | P@10 | Excluded@10 (lower is better) |
  |---|---|---|---|
  | TF-IDF lnc.ltc (course baseline) | 0.462 | 0.342 | 0.214 |
  | BM25F zones on the raw note | 0.465 | 0.324 | 0.232 |
  | Vega, lexical stage | 0.533 | 0.400 | 0.186 |
  | MedCPT alone (topic view) | 0.560 | 0.436 | 0.186 |
  | Vega + MedCPT topic view | 0.591 | 0.468 | 0.192 |
  | + inclusion view | 0.603 | 0.480 | 0.192 |
  | + exclusion view | 0.608 | 0.488 | 0.186 |
  | **Vega-Neural (+ age/sex term)** | **0.611** | **0.494** | **0.182** |

  - **Against BM25F:** Vega-Neural improves nDCG@10 by 31% (paired t-test p = 1.3e-6), and is better for 40 patients and worse for 8.
  - **Against lexical Vega:** p = 0.0004.
  - **Robustness:** 5-fold cross-validation within 2021 also improves (0.553 → 0.595).
  - **Context:** the best team at TREC 2022 scored nDCG@10 0.613 and P@10 0.508 (official overview), so Vega-Neural is on par with the top system.

## Still planned

- Criterion-level neural matching: score each inclusion or exclusion criterion against the patient separately, instead of truncating whole zones at 384 tokens.
- Mapping to medical concepts (UMLS/MeSH) instead of words.
- Fine-tune the cross-encoder on the 2021 judgements, with eligible, excluded and not-relevant as three labels.
- Judging the unjudged trials that Vega retrieves, to remove the pooling bias.
