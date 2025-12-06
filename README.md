# ICS 121 Assignment 3 - Search Engine

A full-featured web search engine implementation with inverted indexing, ranking algorithms, and a web interface.

## Features

- **Inverted Indexing**: Efficient indexing of web documents with support for:
  - Unigram (term) indexing
  - Bigram indexing
  - Trigram indexing
  - Anchor text indexing

- **Advanced Ranking**:
  - TF-IDF (Term Frequency-Inverse Document Frequency) scoring
  - PageRank algorithm for link-based ranking
  - HITS algorithm (Hubs and Authorities)
  - Position-based boosting for query term proximity
  - Important text boosting (title, headings, bold text)
  - Duplicate content detection and filtering

- **Query Processing**:
  - Porter stemming
  - Query tokenization
  - Multi-term query support
  - Relevance scoring

- **Web Interface**:
  - Google-like search interface
  - Real-time search results
  - Relevance scores display
  - Response time tracking

## Project Structure

```
.
├── indexer.py              # Main indexing engine
├── search.py               # Search engine implementation
├── web_search.py           # Flask web application
├── test_performance.py     # Performance testing script
├── test_queries.txt        # Test queries
├── requirements.txt        # Python dependencies
├── templates/
│   └── search.html         # Web interface template
└── [index_dir]/            # Index directory (e.g., dev_index, analyst_index)
    ├── doc_index.json      # Document ID to URL mapping
    ├── final_index.txt     # Main inverted index
    ├── final_bigram_index.txt
    ├── final_trigram_index.txt
    ├── final_anchor_index.txt
    ├── pagerank.json       # PageRank scores
    ├── hits.json           # HITS scores
    ├── duplicates.json     # Duplicate document sets
    └── index_stats.json    # Index statistics
```

## Installation

### Prerequisites

- Python 3.7+
- pip

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (if not already downloaded):
```python
import nltk
nltk.download('punkt')
```

## Usage

### 1. Building an Index

To build an index from a corpus of JSON documents:

```bash
python indexer.py --corpus <corpus_directory> --index-dir <index_directory>
```

**Optional parameters:**
- `--max-docs-in-block`: Maximum documents per block (default: 4000)
- `--max-terms-in-memory`: Maximum terms in memory before flushing (default: 200000)

**Example:**
```bash
python indexer.py --corpus DEV --index-dir dev_index
```

The indexer will:
- Parse HTML content from JSON files
- Extract tokens, important text, and anchor text
- Build inverted indexes (unigram, bigram, trigram, anchor)
- Compute PageRank and HITS scores
- Detect duplicate documents
- Generate index statistics

### 2. Command-Line Search

#### Interactive Mode:
```bash
python search.py --index-dir <index_directory> --interactive
```

#### Single Query:
```bash
python search.py --index-dir <index_directory> --query "your search query"
```

**Example:**
```bash
python search.py --index-dir dev_index --query "machine learning"
```

### 3. Web Interface

Start the Flask web server:

```bash
python web_search.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

The web interface provides:
- Google-like search interface
- Real-time search results
- Relevance scores
- Response time display

**Note:** Update `INDEX_DIR` in `web_search.py` to point to your index directory.

### 4. Performance Testing

Run performance tests with predefined queries:

```bash
python test_performance.py <index_directory>
```

**Example:**
```bash
python test_performance.py dev_index
```

The script will read queries from `test_queries.txt` and report:
- Number of results per query
- Response time for each query
- Overall statistics

## Index Format

### Document Index (`doc_index.json`)
Maps document IDs to URLs:
```json
{
  "0": "https://example.com/page1",
  "1": "https://example.com/page2"
}
```

### Inverted Index (`final_index.txt`)
Tab-separated format:
```
term<TAB>{"postings": [[doc_id, tf, important_tf, positions]], "df": document_frequency}
```

### PageRank (`pagerank.json`)
Document ID to PageRank score mapping.

### HITS (`hits.json`)
Contains hub and authority scores:
```json
{
  "hubs": {"doc_id": hub_score},
  "authorities": {"doc_id": authority_score}
}
```

## Search Algorithm

The search engine uses a multi-factor ranking system:

1. **Base TF-IDF Score**: Calculated for each query term
2. **Important Text Boost**: Terms in titles, headings, and bold text get higher weights
3. **Position Boost**: Terms appearing close together get additional boost
4. **N-gram Scores**: Bigram and trigram matches contribute to relevance
5. **Anchor Text Boost**: Anchor text matches are weighted higher
6. **PageRank Boost**: Documents with higher PageRank scores are boosted
7. **HITS Boost**: Authority scores from HITS algorithm boost relevance
8. **Quality Filtering**: Error pages, raw HTML source files, and invalid URLs are filtered or heavily penalized

## Quality Features

The search engine includes several quality improvements:

- **Error Page Filtering**: Automatically filters out 404, 403, and error pages
- **Raw Source File Filtering**: Excludes server-side files (`.ff`, `.ssi`, `.shtml`, etc.) that display raw HTML
- **Duplicate Detection**: Identifies and handles duplicate content
- **URL Validation**: Validates URLs before including in results
- **Smart Title Extraction**: Generates meaningful titles from URLs, avoiding numeric-only titles

## Performance

The search engine is optimized for large-scale indexing:

- **Block-based Indexing**: Handles large corpora by building indexes in blocks
- **Binary Search**: Uses binary search for large index files (>100MB)
- **Caching**: Implements posting list caching for frequently accessed terms
- **Efficient Merging**: Merges partial indexes efficiently

## Dependencies

- `beautifulsoup4`: HTML parsing
- `nltk`: Natural language processing and stemming
- `flask`: Web interface (optional, for web_search.py)

## File Format

The corpus should consist of JSON files with the following structure:

```json
{
  "url": "https://example.com/page",
  "content": "<html>...</html>"
}
```

## Notes

- The indexer uses Porter stemming for consistent token matching
- Documents are deduplicated using content hashing and SimHash
- The search engine supports both exact and approximate matching through n-grams
- Large indexes are automatically handled with binary search optimization

## License

This project is part of ICS 121 (Information Retrieval) coursework at UC Irvine.
