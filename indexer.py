import os
import re
import json
import argparse
import math
import sys
import hashlib
from collections import Counter, defaultdict
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer

# -------------------- Tokenization & HTML parsing -------------------- #

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


class HTMLTokenizer:
    """
    Parse HTML and produce:
      - all_tokens: every alphanumeric token in the page
      - important_tokens: tokens from title, h1, h2, h3, b, strong
    Both lists are Porter-stemmed and lowercased.
    """

    def __init__(self):
        self.stemmer = PorterStemmer()

    def _tokenize_text(self, text):
        """Return a list of stemmed tokens from raw text."""
        tokens = []
        if not text:
            return tokens
        for match in TOKEN_PATTERN.finditer(text):
            raw = match.group(0)
            if not raw:
                continue
            raw = raw.lower()
            stem = self.stemmer.stem(raw)
            tokens.append(stem)
        return tokens

    def extract_tokens(self, html_content):
        if not html_content:
            return [], [], [], {}

        try:
            soup = BeautifulSoup(html_content, "html.parser")
        except Exception:
            tokens = self._tokenize_text(html_content)
            positions = {i: [i] for i, _ in enumerate(tokens)}
            return tokens, [], [], positions

        important_chunks = []
        if soup.title is not None:
            important_chunks.append(soup.title.get_text(separator=" ", strip=True))
        for tag_name in ["h1", "h2", "h3", "b", "strong"]:
            for tag in soup.find_all(tag_name):
                important_chunks.append(tag.get_text(separator=" ", strip=True))

        important_text = " ".join(important_chunks)
        important_tokens = self._tokenize_text(important_text)

        all_text = soup.get_text(separator=" ", strip=True)
        all_tokens = self._tokenize_text(all_text)

        positions = defaultdict(list)
        for idx, token in enumerate(all_tokens):
            positions[token].append(idx)

        anchor_texts = []
        for link in soup.find_all("a", href=True):
            anchor_text = link.get_text(separator=" ", strip=True)
            if anchor_text:
                anchor_tokens = self._tokenize_text(anchor_text)
                anchor_texts.extend(anchor_tokens)

        return all_tokens, important_tokens, anchor_texts, dict(positions)


# -------------------- Inverted index builder -------------------- #


class InvertedIndexer:
    def __init__(
        self,
        index_dir,
        max_docs_in_block=4000,
        max_terms_in_memory=200000,
    ):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)

        self.partial_index = defaultdict(list)
        self.partial_bigram_index = defaultdict(list)
        self.partial_trigram_index = defaultdict(list)
        self.partial_anchor_index = defaultdict(list)

        self.max_docs_in_block = max_docs_in_block
        self.max_terms_in_memory = max_terms_in_memory

        self.docs_in_current_block = 0
        self.partial_file_count = 0

        self.doc_count = 0
        self.docid_to_url = {}
        self.url_to_docid = {}
        self.doc_content_hash = {}
        self.doc_simhash = {}
        self.link_graph = defaultdict(set)
        self.url_to_base = {}

        self.tokenizer = HTMLTokenizer()

        self.unique_terms = 0
        self.final_index_path = os.path.join(self.index_dir, "final_index.txt")
        self.final_bigram_index_path = os.path.join(self.index_dir, "final_bigram_index.txt")
        self.final_trigram_index_path = os.path.join(self.index_dir, "final_trigram_index.txt")
        self.final_anchor_index_path = os.path.join(self.index_dir, "final_anchor_index.txt")
        self.doc_index_path = os.path.join(self.index_dir, "doc_index.json")
        self.stats_path = os.path.join(self.index_dir, "index_stats.json")
        self.duplicates_path = os.path.join(self.index_dir, "duplicates.json")
        self.pagerank_path = os.path.join(self.index_dir, "pagerank.json")
        self.hits_path = os.path.join(self.index_dir, "hits.json")

    # ---------- corpus traversal ---------- #

    def iter_corpus_files(self, corpus_root):
        """
        Iterate over all JSON files in the corpus directory tree.
        Each file corresponds to one crawled web page.
        """
        for root, dirs, files in os.walk(corpus_root):
            for name in files:
                if not name.lower().endswith(".json"):
                    continue
                yield os.path.join(root, name)

    # ---------- document processing ---------- #

    def compute_simhash(self, tokens, hash_bits=64):
        v = [0] * hash_bits
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            for i in range(hash_bits):
                if h & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1
        fingerprint = 0
        for i in range(hash_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint

    def hamming_distance(self, h1, h2):
        return bin(h1 ^ h2).count('1')

    def add_document(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read {file_path}: {e}", file=sys.stderr)
            return

        url = data.get("url", file_path)
        if "#" in url:
            url = url.split("#")[0]
        html_content = data.get("content", "")

        content_hash = hashlib.md5(html_content.encode('utf-8', errors='ignore')).hexdigest()
        if content_hash in self.doc_content_hash:
            return

        doc_id = self.doc_count
        self.doc_count += 1
        self.docid_to_url[doc_id] = url
        self.url_to_docid[url] = doc_id
        self.doc_content_hash[content_hash] = doc_id

        all_tokens, important_tokens, anchor_tokens, positions = self.tokenizer.extract_tokens(html_content)

        if not all_tokens:
            return

        simhash = self.compute_simhash(all_tokens)
        self.doc_simhash[doc_id] = simhash

        tf_counter = Counter(all_tokens)
        important_counter = Counter(important_tokens)
        anchor_counter = Counter(anchor_tokens)

        for term, tf in tf_counter.items():
            important_tf = important_counter.get(term, 0)
            pos_list = positions.get(term, [])
            self.partial_index[term].append((doc_id, tf, important_tf, pos_list))

        for i in range(len(all_tokens) - 1):
            bigram = f"{all_tokens[i]}_{all_tokens[i+1]}"
            self.partial_bigram_index[bigram].append((doc_id, 1))

        for i in range(len(all_tokens) - 2):
            trigram = f"{all_tokens[i]}_{all_tokens[i+1]}_{all_tokens[i+2]}"
            self.partial_trigram_index[trigram].append((doc_id, 1))

        for anchor_term, anchor_tf in anchor_counter.items():
            self.partial_anchor_index[anchor_term].append((doc_id, anchor_tf))

        try:
            soup = BeautifulSoup(html_content, "html.parser")
            base_url = url
            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                if href:
                    absolute_url = urljoin(base_url, href)
                    if "#" in absolute_url:
                        absolute_url = absolute_url.split("#")[0]
                    if absolute_url in self.url_to_docid:
                        target_doc_id = self.url_to_docid[absolute_url]
                        self.link_graph[doc_id].add(target_doc_id)
        except Exception:
            pass

        self.docs_in_current_block += 1

        if (
            self.docs_in_current_block >= self.max_docs_in_block
            or len(self.partial_index) >= self.max_terms_in_memory
        ):
            self.flush_partial_index()

    # ---------- partial index flushing ---------- #

    def flush_partial_index(self):
        if not self.partial_index:
            return

        self.partial_file_count += 1
        base_filename = f"partial_{self.partial_file_count:03d}"

        print(f"[INFO] Flushing partial index #{self.partial_file_count}")

        with open(os.path.join(self.index_dir, f"{base_filename}.txt"), "w", encoding="utf-8") as f:
            for term in sorted(self.partial_index.keys()):
                postings = self.partial_index[term]
                obj = {"postings": postings}
                line = term + "\t" + json.dumps(obj) + "\n"
                f.write(line)

        with open(os.path.join(self.index_dir, f"{base_filename}_bigram.txt"), "w", encoding="utf-8") as f:
            for term in sorted(self.partial_bigram_index.keys()):
                postings = self.partial_bigram_index[term]
                obj = {"postings": postings}
                line = term + "\t" + json.dumps(obj) + "\n"
                f.write(line)

        with open(os.path.join(self.index_dir, f"{base_filename}_trigram.txt"), "w", encoding="utf-8") as f:
            for term in sorted(self.partial_trigram_index.keys()):
                postings = self.partial_trigram_index[term]
                obj = {"postings": postings}
                line = term + "\t" + json.dumps(obj) + "\n"
                f.write(line)

        with open(os.path.join(self.index_dir, f"{base_filename}_anchor.txt"), "w", encoding="utf-8") as f:
            for term in sorted(self.partial_anchor_index.keys()):
                postings = self.partial_anchor_index[term]
                obj = {"postings": postings}
                line = term + "\t" + json.dumps(obj) + "\n"
                f.write(line)

        self.partial_index.clear()
        self.partial_bigram_index.clear()
        self.partial_trigram_index.clear()
        self.partial_anchor_index.clear()
        self.docs_in_current_block = 0

    # ---------- merging partial indexes ---------- #

    def merge_partials(self):
        import glob

        def merge_index_files(pattern, output_path, index_name):
            partial_paths = sorted(glob.glob(pattern))
            if not partial_paths:
                print(f"[WARN] No {index_name} partial files found.")
                return 0

            print(f"[INFO] Merging {len(partial_paths)} {index_name} partial files...")
            fps = [open(path, "r", encoding="utf-8") for path in partial_paths]
            current_lines = []
            for fp in fps:
                line = fp.readline()
                current_lines.append(line.rstrip("\n") if line else None)

            unique_terms = 0
            with open(output_path, "w", encoding="utf-8") as out:
                while True:
                    candidates = []
                    for i, line in enumerate(current_lines):
                        if not line:
                            continue
                        try:
                            term, payload = line.split("\t", 1)
                            data = json.loads(payload)
                            postings = data.get("postings", [])
                            candidates.append((term, i, postings))
                        except ValueError:
                            continue

                    if not candidates:
                        break

                    min_term = min(t for (t, _, _) in candidates)
                    merged_postings = []

                    for term, file_idx, postings in candidates:
                        if term != min_term:
                            continue
                        merged_postings.extend(postings)
                        next_line = fps[file_idx].readline()
                        current_lines[file_idx] = next_line.rstrip("\n") if next_line else None

                    merged_postings.sort(key=lambda x: x[0])
                    df = len(merged_postings)
                    out_obj = {"df": df, "postings": merged_postings}
                    out_line = min_term + "\t" + json.dumps(out_obj) + "\n"
                    out.write(out_line)
                    unique_terms += 1

            for fp in fps:
                fp.close()

            for path in partial_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

            print(f"[INFO] {index_name} merge finished. Unique terms: {unique_terms}")
            return unique_terms

        pattern_bigram = os.path.join(self.index_dir, "partial_*_bigram.txt")
        pattern_trigram = os.path.join(self.index_dir, "partial_*_trigram.txt")
        pattern_anchor = os.path.join(self.index_dir, "partial_*_anchor.txt")
        pattern_main = os.path.join(self.index_dir, "partial_[0-9]*.txt")

        merge_index_files(pattern_bigram, self.final_bigram_index_path, "bigram")
        merge_index_files(pattern_trigram, self.final_trigram_index_path, "trigram")
        merge_index_files(pattern_anchor, self.final_anchor_index_path, "anchor")
        self.unique_terms = merge_index_files(pattern_main, self.final_index_path, "main")

    # ---------- stats & metadata ---------- #

    def write_doc_index(self):
        """Write mapping from doc_id to URL."""
        with open(self.doc_index_path, "w", encoding="utf-8") as f:
            json.dump(self.docid_to_url, f, indent=2)

    def compute_index_size_kb(self):
        """Compute total size (KB) of main index files on disk."""
        total_bytes = 0
        for path in [self.final_index_path, self.doc_index_path, self.stats_path]:
            if os.path.exists(path):
                total_bytes += os.path.getsize(path)
        return round(total_bytes / 1024.0, 2)

    def detect_duplicates(self):
        exact_duplicates = defaultdict(list)
        for content_hash, doc_id in self.doc_content_hash.items():
            exact_duplicates[content_hash].append(doc_id)

        near_duplicates = defaultdict(list)
        doc_ids = list(self.doc_simhash.keys())
        for i, doc_id1 in enumerate(doc_ids):
            for doc_id2 in doc_ids[i+1:]:
                if self.hamming_distance(self.doc_simhash[doc_id1], self.doc_simhash[doc_id2]) <= 3:
                    near_duplicates[doc_id1].append(doc_id2)

        duplicates = {
            "exact": {str(k): v for k, v in exact_duplicates.items() if len(v) > 1},
            "near": {str(k): v for k, v in near_duplicates.items() if v}
        }
        with open(self.duplicates_path, "w", encoding="utf-8") as f:
            json.dump(duplicates, f, indent=2)
        return duplicates

    def compute_pagerank(self, damping=0.85, max_iter=100, tol=1e-6):
        num_docs = self.doc_count
        if num_docs == 0:
            return {}

        outlinks = {doc_id: len(links) for doc_id, links in self.link_graph.items()}
        pr = {doc_id: 1.0 / num_docs for doc_id in range(num_docs)}

        for iteration in range(max_iter):
            new_pr = {}
            for doc_id in range(num_docs):
                new_pr[doc_id] = (1 - damping) / num_docs
                for src_doc, targets in self.link_graph.items():
                    if doc_id in targets and outlinks[src_doc] > 0:
                        new_pr[doc_id] += damping * pr[src_doc] / outlinks[src_doc]

            diff = sum(abs(new_pr[doc_id] - pr[doc_id]) for doc_id in range(num_docs))
            pr = new_pr
            if diff < tol:
                break

        with open(self.pagerank_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in pr.items()}, f, indent=2)
        return pr

    def compute_hits(self, max_iter=100, tol=1e-6):
        num_docs = self.doc_count
        if num_docs == 0:
            return {}, {}

        hubs = {doc_id: 1.0 for doc_id in range(num_docs)}
        authorities = {doc_id: 1.0 for doc_id in range(num_docs)}

        inlinks = defaultdict(set)
        for src, targets in self.link_graph.items():
            for target in targets:
                inlinks[target].add(src)

        for iteration in range(max_iter):
            new_authorities = {}
            new_hubs = {}

            norm_a = 0.0
            for doc_id in range(num_docs):
                auth = sum(hubs[src] for src in inlinks[doc_id])
                new_authorities[doc_id] = auth
                norm_a += auth * auth

            norm_a = math.sqrt(norm_a) if norm_a > 0 else 1.0
            for doc_id in range(num_docs):
                new_authorities[doc_id] /= norm_a

            norm_h = 0.0
            for doc_id in range(num_docs):
                hub = sum(new_authorities[target] for target in self.link_graph[doc_id])
                new_hubs[doc_id] = hub
                norm_h += hub * hub

            norm_h = math.sqrt(norm_h) if norm_h > 0 else 1.0
            for doc_id in range(num_docs):
                new_hubs[doc_id] /= norm_h

            diff_a = sum(abs(new_authorities[doc_id] - authorities[doc_id]) for doc_id in range(num_docs))
            diff_h = sum(abs(new_hubs[doc_id] - hubs[doc_id]) for doc_id in range(num_docs))

            authorities = new_authorities
            hubs = new_hubs

            if diff_a < tol and diff_h < tol:
                break

        hits_data = {
            "hubs": {str(k): v for k, v in hubs.items()},
            "authorities": {str(k): v for k, v in authorities.items()}
        }
        with open(self.hits_path, "w", encoding="utf-8") as f:
            json.dump(hits_data, f, indent=2)
        return hubs, authorities

    def compute_index_size_kb(self):
        total_bytes = 0
        index_files = [
            self.final_index_path, self.final_bigram_index_path,
            self.final_trigram_index_path, self.final_anchor_index_path,
            self.doc_index_path, self.stats_path, self.duplicates_path,
            self.pagerank_path, self.hits_path
        ]
        for path in index_files:
            if os.path.exists(path):
                total_bytes += os.path.getsize(path)
        return round(total_bytes / 1024.0, 2)

    def write_stats(self):
        index_size_kb = self.compute_index_size_kb()
        stats = {
            "num_documents": self.doc_count,
            "num_unique_terms": self.unique_terms,
            "index_size_kb": index_size_kb,
        }
        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print("========== INDEX STATS ==========")
        print(f"Number of indexed documents : {self.doc_count}")
        print(f"Number of unique tokens      : {self.unique_terms}")
        print(f"Index size on disk (KB)      : {index_size_kb}")
        print("=================================")

    # ---------- public API ---------- #

    def build(self, corpus_root):
        print(f"[INFO] Indexing corpus at: {corpus_root}")
        for file_path in self.iter_corpus_files(corpus_root):
            self.add_document(file_path)

        if self.partial_index:
            self.flush_partial_index()

        self.merge_partials()
        self.write_doc_index()

        print("[INFO] Detecting duplicates...")
        self.detect_duplicates()

        print("[INFO] Computing PageRank...")
        self.compute_pagerank()

        print("[INFO] Computing HITS...")
        self.compute_hits()

        self.write_stats()


# -------------------- CLI -------------------- #


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assignment 3 – Milestone 1: Inverted Index Builder"
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to corpus root directory (e.g., ./developer or ./analyst)",
    )
    parser.add_argument(
        "--index-dir",
        required=True,
        help="Directory where the index files will be written",
    )
    parser.add_argument(
        "--max-docs-in-block",
        type=int,
        default=4000,
        help="Max number of documents per in-memory block before flushing",
    )
    parser.add_argument(
        "--max-terms-in-memory",
        type=int,
        default=200000,
        help="Max number of distinct terms in memory before flushing",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    corpus_root = os.path.abspath(args.corpus)
    index_dir = os.path.abspath(args.index_dir)

    if not os.path.isdir(corpus_root):
        print(f"[ERROR] Corpus directory does not exist: {corpus_root}")
        sys.exit(1)

    indexer = InvertedIndexer(
        index_dir=index_dir,
        max_docs_in_block=args.max_docs_in_block,
        max_terms_in_memory=args.max_terms_in_memory,
    )
    indexer.build(corpus_root)


if __name__ == "__main__":
    main()
