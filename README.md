# CSAI 422 Assignment 4: Advanced RAG on PopQA

This repository contains my implementation of Assignment 4 for CSAI 422. The project builds an advanced Retrieval-Augmented Generation (RAG) pipeline for factual question answering using the PopQA dataset.

## Project Overview

The system compares several retrieval and generation configurations:

- Dense retrieval baseline
- Dense retrieval with query expansion
- Hybrid retrieval using BM25 and dense search
- Hybrid retrieval with cross-encoder reranking
- Citation-grounded answer generation
- Self-reflective RAG for answer checking and revision

The goal is to study how retrieval, ranking, grounding, and reflection affect factual question-answering performance.

## Dataset

The project uses the PopQA dataset from Hugging Face.  
A reproducible evaluation subset of 100 questions was selected using `random_state = 42`.

The main fields used were:

- `question`: natural-language question
- `obj` / `possible_answers`: gold answer and accepted aliases
- `subj`: subject entity
- `prop`: relation/property
- `s_aliases`: subject aliases used for query expansion
- `s_wiki_title`: Wikipedia title used for corpus construction

## Retrieval Corpus

The retrieval corpus was built from Wikipedia summaries corresponding to PopQA subject pages, with additional distractor pages.  
Each passage preserves:

- passage ID
- title
- source URL
- passage text

## Methods

### 1. Dense Retrieval

The dense baseline uses `sentence-transformers/all-MiniLM-L6-v2` to embed questions and passages.  
A FAISS index is used for similarity search.

### 2. Query Expansion

Queries are expanded using PopQA metadata such as subject names, aliases, and relation keywords.

### 3. Hybrid Search

Hybrid retrieval combines dense retrieval with BM25 lexical search.  
Reciprocal Rank Fusion (RRF) is used to merge the rankings.

### 4. Reranking

A cross-encoder reranker, `cross-encoder/ms-marco-MiniLM-L-6-v2`, is applied to the top hybrid candidates.

### 5. Citation-Grounded Answer Generation

The best retrieval pipeline is connected to an LLM.  
Answers are generated only from retrieved passages and include citations such as `[P1]` and `[P2]`.

### 6. Self-Reflective RAG

A reflection step checks whether the answer is grounded, complete, and properly cited.  
It either keeps the answer, revises it, or returns insufficient evidence.

## Results Summary

The best retrieval configuration was Hybrid Retrieval + Reranking.

| System | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| Dense baseline | 0.27 | 0.30 | 0.34 | 0.295000 |
| Dense + query expansion | 0.28 | 0.32 | 0.34 | 0.303333 |
| Hybrid RRF | 0.26 | 0.33 | 0.35 | 0.296167 |
| Hybrid + reranker | 0.32 | 0.35 | 0.36 | 0.334167 |

Self-reflection did not change retrieval scores because it occurs after retrieval, but it improved answer reliability by checking grounding and citation quality.

## Files

- `hana_assign4.ipynb` — main assignment notebook
- `hana report .docx` or `CSAI422_Assignment4_Report.pdf` — final report
- `requirements.txt` — required Python libraries
- `retrieval_comparison.csv` — retrieval metric comparison
- `grounded_qa_examples.csv` — citation-grounded QA examples
- `failure_analysis.csv` — error analysis examples
- `final_comparison.csv` — final system comparison

## How to Run

1. Open the notebook in Google Colab.
2. Install the required libraries:

```bash
pip install datasets sentence-transformers faiss-cpu rank-bm25 pandas numpy scikit-learn tqdm transformers groq wikipedia
Add the Groq API key if running the answer generation and reflection sections:
import os
os.environ["GROQ_API_KEY"] = "your_api_key_here"
Run all notebook cells from top to bottom.
The notebook will generate retrieval metrics, grounded QA examples, failure analysis, and final comparison tables.

Requirements

Main libraries used:

Python 3.10+
datasets
sentence-transformers
faiss-cpu
rank-bm25
pandas
numpy
scikit-learn
transformers
groq
wikipedia
