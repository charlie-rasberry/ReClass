# RECLASS: Multi-Task Deep Learning for App Review Classification

**COMP6013 | Oxford Brookes University | 2025-26**

---

# README not finished

## Overview

RECLASS is a multitask learning system which uses a shared multilingual transformer encoder with task-specific heads and single-task implementations for optional comparison.

| Task | Output | Classes |
|------|--------|---------|
| Bug Report Detection | Binary | Yes / No |
| Feature Request Detection | Binary | Yes / No |
| Aspect Classification | Multi-class | Driver, App, Pricing, Service, Payment, General |
| Aspect Sentiment | Multi-class | Positive, Neutral, Negative |

## Dataset

- **Source**: [Uber Customer Reviews (Kaggle)](https://www.kaggle.com/datasets/khushipitroda/ola-vs-uber-play-store-reviews)
- **Original size**: ~1.07M Reviews
- **After Preprocessing**: ~495K Reviews
- **Annotation subsets**: 5,000 from the original distribution, 5,000 from a keyword boosted sample

## Preprocessing Steps

- Removed URLS and emails
- Normalised text and punctuation
- Removed duplicate reviews
- Filtered reviews less than 5 words

- Output sets
    -   Original: matches the original distribution of the raw dataset
    -   Boosted: oversamples bug reports and feature requests using keyword heuristics

## Model

- Encoder: XLM-RoBERTa (large multilingual transformer model)
- Architecture:
    - Shared encoder
    - Task-specific classification heads
- Training setups:
    - MTL (Multitask learning)
    - STL (Single-task learning)

Class weights are applied to reduce imbalance effects.

## Repository Structure

.
├── data
│   └── processed
│       ├── boosted_test.csv
│       ├── boosted_train.csv
│       ├── boosted_val.csv
│       ├── original_test.csv
│       ├── original_train.csv
│       ├── original_val.csv
│       └── review.csv
├── notebooks/
│   
├── outputs
│   └── figures/
├── README.md
├── architecture.png
└── src
    ├── dataset.py
    ├── evaluate.py
    ├── infer.py
    ├── model.py
    ├── multitag.py
    ├── preprocess.py
    ├── sampler.py
    └── train.py

## Results

Evaluation includes Precision, Recall, Macro F1, Confusion matrices and confidence analysis.

Results and summaries are found in outputs/*.json and outputs/figures/

## Installation

```
# Create conda environment
conda create -n reclass python=3.11 
conda activate reclass
```

```
# Install dependencies
conda install --file requirements.txt
```

## Usage

#### Train Model

```
python src/train.py --mode mtl --dataset original
```

#### Evaluate Model

```
python src/evaluate.py --mode mtl --dataset original --model_path <model>.pt
```

#### Run Inference

```
python src/infer.py --mode mtl --model_path <model>.pt --dataset review
```

## Notes

- The same tokenizer is used across training, evaluation and inference to ensure consistency
- Sampling and preprocessing choices are documented further in src files and dissertation

---

*Last updated: January 2025*
