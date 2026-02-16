# RECLASS: Multi-Task Deep Learning for App Review Classification

**COMP6013 | Oxford Brookes University | 2025-26**

---

## Project Overview

RECLASS is a multi-task learning system which uses a shared BERT encoder with task-specific classification heads.

| Task | Output | Classes |
|------|--------|---------|
| Bug Report Detection | Binary | Yes / No |
| Feature Request Detection | Binary | Yes / No |
| Aspect Classification | Multi-class | Driver, App, Pricing, Service, Payment, General |
| Aspect Sentiment | Multi-class | Positive, Neutral, Negative |

## Dataset

- **Source**: [Uber Customer Reviews (Kaggle)](https://www.kaggle.com/datasets/khushipitroda/ola-vs-uber-play-store-reviews)
- **Original size**: 1,069,616 reviews
- **Cleaned size**: 495,036 reviews (after removing short/duplicate reviews)
- **Annotation target**: 5,000 manually labelled reviews

## Repository Structure

```
## Repository Structure

6013/
    README.md
    .gitignore
    data/
        uber_reviews.csv           # Raw dataset
        uber_reviews_cleaned.csv   # Preprocessed reviews
        uber_reviews_sampled.csv   # Stratified sample for annotation
        uber_reviews_tagged.csv    # Annotated reviews (in progress)
    notebooks/
        preprocessing_uber.ipynb   # Preprocessing analysis
        uber_cleaned.ipynb         # Cleaned data verification
    src/
        preprocess.py              # Text cleaning and filtering pipeline
        sampler.py                 # Stratified sampling strategies
        multitag.py                # GUI annotation tool
        train.py                   # Model training (in progress)
        infer.py                   # Inference pipeline (in progress)
        outputs/
            figures/
```

## Current Progress

- Manual annotation of 5,000 reviews
- BERT baseline implementation
- Multi-task model architecture
- Training and evaluation
- Comparative analysis (MTL vs single-task)
- Final report and presentation

## Installation

```
# Clone repository
...
# Create conda environment
...
# Install dependencies
...requirements.txt
```

## Usage
## References
## Licenses

---

*Last updated: January 2025*
