# OpenFDA Drug Label Data Processing and Vector Search Testing

This directory contains scripts for processing OpenFDA drug label data and testing vector search capabilities using FAISS.

## Overview

The project processes drug label data from OpenFDA and creates test queries to evaluate the performance of different vector storage systems, with a focus on FAISS (Facebook AI Similarity Search).

## Files Description

- `data_extractor.py` - Main script for extracting and cleaning drug label data
- `test_queries.py` - Script for generating realistic test queries
- `FAISS_test.py` - FAISS vector search testing script
- `drug-label-0001-of-0013.json` - Original OpenFDA drug label dataset
- `extracted_drug_documents.json` - Processed drug documents
- `extracted_drug_documents_texts.txt` - Plain text version of extracted documents
- `test_queries.json` - Generated test queries with ground truth
- `faiss_test_results.json` - FAISS test results

## Prerequisites

Before running the scripts, install the required dependencies:

```bash
pip install faiss-cpu sentence-transformers numpy
```

## Data Extraction Process

### Step 1: Extract Drug Documents

Run the data extraction script to process the raw OpenFDA data:

```bash
python data_extractor.py
```

This script will:
- Load the original OpenFDA JSON file (`drug-label-0001-of-0013.json`)
- Extract key drug information including:
  - Drug names (brand and generic)
  - Active and inactive ingredients
  - Indications and usage
  - Dosage and administration
  - Warnings and safety information
  - Pregnancy information
  - Overdosage information
- Clean and normalize text data
- Create structured documents for vector search
- Save processed data to `extracted_drug_documents.json`
- Generate a plain text file for easy inspection

### Output Files

After extraction, you'll have:
- `extracted_drug_documents.json` - Structured drug documents with metadata
- `extracted_drug_documents_texts.txt` - Human-readable text format

### Data Statistics

The extraction process typically processes 1,000 drug records and provides statistics including:
- Total number of documents
- Average document length
- Document feature statistics (ingredients, warnings, dosage info, etc.)

## Test Query Generation

### Step 2: Generate Test Queries

Create realistic test queries for evaluating vector search performance:

```bash
python test_queries.py
```

This script generates 4 non-trivial test queries:

1. **Cardiovascular Disease Treatment** (Medium difficulty)
   - Query: "What medications are used to treat high blood pressure and cardiovascular conditions?"
   - Focus: Therapeutic use queries

2. **Pregnancy-Safe Acetaminophen** (High difficulty)
   - Query: "Find drugs that contain acetaminophen and are safe for pregnant women"
   - Focus: Safety and ingredient queries

3. **Antidepressant Side Effects** (High difficulty)
   - Query: "What are the side effects and warnings for medications used to treat depression and anxiety?"
   - Focus: Safety warnings and side effects

4. **OTC Pain Reliever Safety** (Very high difficulty)
   - Query: "Find over-the-counter pain relievers that can be taken with food and have minimal drug interactions"
   - Focus: Complex safety and interaction queries

### Ground Truth Generation

The script automatically generates ground truth labels based on keyword matching:
- Cardiovascular keywords: "blood pressure", "cardiovascular", "heart", "hypertension"
- Safety keywords: "pregnant", "pregnancy", "safe", "warning"
- Drug interaction keywords: "interaction", "contraindication", "avoid"

## FAISS Vector Search Testing

### Step 3: Run FAISS Tests

Execute the FAISS vector search testing:

```bash
python FAISS_test.py
```

This script will:
- Load the extracted drug documents
- Load the test queries
- Initialize Sentence Transformer model (`all-MiniLM-L6-v2`)
- Generate document embeddings
- Create FAISS index with inner product similarity
- Test each query against the vector index
- Calculate performance metrics (Precision, Recall, F1-Score)
- Generate detailed results

### Performance Metrics

The test evaluates:
- **Precision**: Percentage of retrieved documents that are relevant
- **Recall**: Percentage of relevant documents that were retrieved
- **F1-Score**: Harmonic mean of precision and recall
- **Query Time**: Time taken to process each query

### Expected Results

Typical performance ranges:
- **Query 1 (Cardiovascular)**: P=0.4, R=0.4, F1=0.4
- **Query 2 (Acetaminophen)**: P=0.2, R=0.2, F1=0.2
- **Query 3 (Antidepressants)**: P=0.1, R=0.1, F1=0.1
- **Query 4 (OTC Pain Relievers)**: P=0.2, R=0.2, F1=0.2

## Understanding the Results

### Output Files

- `faiss_test_results.json` - Detailed test results with retrieved documents and scores
- Console output shows performance summary and top retrieved documents

### Result Interpretation

1. **High Precision**: Retrieved documents are highly relevant to the query
2. **High Recall**: Most relevant documents are successfully retrieved
3. **Balanced F1-Score**: Good balance between precision and recall
4. **Query Time**: Efficiency of the vector search system

### Sample Output

```
Query query_1: Cardiovascular disease treatment drug query
Difficulty: medium
Query: What medications are used to treat high blood pressure and cardiovascular conditions?
Performance metrics:
  - Precision: 0.400
  - Recall: 0.400
  - F1 Score: 0.400
  - Query time: 1.105 seconds
  - Relevant documents retrieved: 4/10

Retrieval results (top 5):
  1. Benazepril Hydrochloride (BENAZEPRIL HYDROCHLORIDE) - Score: 0.625
     Preview: Drug Name: Benazepril Hydrochloride (BENAZEPRIL HYDROCHLORIDE)...
```

## Customization

### Modifying Test Queries

Edit `test_queries.py` to add new test queries:
- Add new query dictionaries to the `queries` list
- Update the ground truth generation logic in `create_ground_truth()`
- Adjust keyword matching for different query types

### Adjusting Data Processing

Modify `data_extractor.py` to:
- Change the number of processed records (default: 1,000)
- Add new fields to extract from the OpenFDA data
- Modify text cleaning and normalization rules

### FAISS Configuration

Customize `FAISS_test.py` to:
- Use different embedding models
- Adjust the number of retrieved documents (default: 10)
- Change similarity metrics
- Modify performance evaluation criteria

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce the number of processed documents in `data_extractor.py`
2. **Slow Performance**: Use a smaller embedding model or reduce batch sizes
3. **Low Accuracy**: Check ground truth generation logic or try different embedding models

### Dependencies

Ensure all required packages are installed:
```bash
pip install faiss-cpu sentence-transformers numpy json pathlib
```

## Next Steps

This framework can be extended to test other vector storage systems:
- Qdrant
- Weaviate
- Elasticsearch
- Pinecone

The modular design allows easy integration of additional vector databases for comparative analysis.

## Data Source

The original data comes from the OpenFDA Drug Label API:
- **Source**: https://open.fda.gov/apis/drug/label/
- **License**: Public domain
- **Disclaimer**: This data is for research purposes only and should not be used for medical decisions


