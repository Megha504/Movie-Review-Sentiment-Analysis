# Movie-Review-Sentiment-Analysis
[![Open in nbviewer](https://img.shields.io/badge/Render-nbviewer-orange?style=for-the-badge&logo=jupyter)](https://nbviewer.org/github/Megha504/Movie-Review-Sentiment-Analysis/blob/main/Movie_Review_Sentiment_Analysis.ipynb)

> **Important Note on Notebook Rendering:** This repository contains the complete project pipeline embedded within a single Google Colab Jupyter Notebook (`Movie_Review_Sentiment_Analysis.ipynb`). Because GitHub's native viewer cannot render interactive HTML elements or live user interfaces (such as the Gradio web interface block at the end), please click the **Render nbviewer** badge above to view the entire notebook with all working visualizations and metrics fully rendered.

## Project Overview
This project delivers a complete, end-to-end Natural Language Processing (NLP) pipeline designed to automatically analyze textual movie reviews and classify them into **Positive** or **Negative** sentiments. Built using the standard 50,000 IMDB Movie Reviews dataset, the core focus is to implement a highly optimized, fast, and mathematically transparent classification engine using text normalization frameworks and a tuned **Logistic Regression** model.

To provide seamless accessibility, a web-based graphical user interface (GUI) is integrated directly at the bottom of the execution pipeline using **Gradio**, enabling real-time review evaluations.

### Key Objectives
* Implement a robust text-preprocessing pipeline to thoroughly clean raw text inputs.
* Transform arbitrary-length text strings into dense vector representations using mathematical Term Frequency-Inverse Document Frequency (**TF-IDF**).
* Construct a lightweight, rapidly converging **Logistic Regression** classifier capable of delivering top-tier predictive stability.
* Formulate comprehensive validation checks utilizing key metrics: Precision, Recall, F1-Score, and Confusion Matrix plots.
* Build an abstract web interface module via Gradio to provide real-time inference testing capabilities.

## Methodology & Pipeline Architecture

```text
[Raw Movie Review Text] 
         │
         ▼
[Text Preprocessing] ──► (HTML Stripping ➔ Special Char Removal ➔ Stemming)
         │
         ▼
[Feature Extraction] ──► (TF-IDF Vectorization - n-gram Range (1,3))
         │
         ▼
[Classification Engine] ─► (Trained Logistic Regression)
         │
         ▼
[Deployment Layer] ────► (Gradio Web Interface Output)

```
## Repository Contents & Local Setup
Movie_Review_Sentiment_Analysis.ipynb — The primary Google Colab Notebook encompassing data extraction, cleaning pipelines, training loops, evaluation graphics, and the interface runtime.

If you wish to download this notebook and execute it natively on your machine, run these setup steps in your local terminal

1. Clone the repository
```text
git clone https://github.com/Megha504/Movie-Review-Sentiment-Analysis.git
```

2. Access the project root folder
```text
cd Movie-Review-Sentiment-Analysis
```
4. Initialize your python isolated virtual environment
```text
python -m venv .venv
```
5. Activate the virtual environment:
On Windows Git Bash/Command Prompt:
```text
source .venv/Scripts/activate
```
6. Install foundational project dependencies
``` text
pip install scikit-learn nltk pandas numpy gradio matplotlib seaborn
```
7. Initialize runtime support datasets for NLTK text filters
```text
python -c "import nltk; nltk.download('stopwords')"
```
