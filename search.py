import os
import json
import re
import math
import time
import argparse
from collections import defaultdict
from nltk.stem.porter import PorterStemmer

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


class SearchEngine:
    def __init__(self, index_dir):
        self.index_dir = index_dir
        self.stemmer = PorterStemmer()
        
        self.doc_index_path = os.path.join(index_dir, "doc_index.json")
        self.final_index_path = os.path.join(index_dir, "final_index.txt")
        self.final_bigram_index_path = os.path.join(index_dir, "final_bigram_index.txt")
        self.final_trigram_index_path = os.path.join(index_dir, "final_trigram_index.txt")
        self.final_anchor_index_path = os.path.join(index_dir, "final_anchor_index.txt")
        self.pagerank_path = os.path.join(index_dir, "pagerank.json")
        self.hits_path = os.path.join(index_dir, "hits.json")
        self.duplicates_path = os.path.join(index_dir, "duplicates.json")
        
        self.docid_to_url = {}
        self.total_docs = 0
        self.pagerank = {}
        self.hubs = {}
        self.authorities = {}
        self.duplicate_sets = set()
        self._posting_cache = {}
        
        self._load_metadata()
    
    def _load_metadata(self):
        with open(self.doc_index_path, "r", encoding="utf-8") as f:
            self.docid_to_url = {int(k): v for k, v in json.load(f).items()}
        self.total_docs = len(self.docid_to_url)
        
        if os.path.exists(self.pagerank_path):
            with open(self.pagerank_path, "r", encoding="utf-8") as f:
                self.pagerank = {int(k): v for k, v in json.load(f).items()}
        
        if os.path.exists(self.hits_path):
            with open(self.hits_path, "r", encoding="utf-8") as f:
                hits_data = json.load(f)
                self.hubs = {int(k): v for k, v in hits_data.get("hubs", {}).items()}
                self.authorities = {int(k): v for k, v in hits_data.get("authorities", {}).items()}
        
        if os.path.exists(self.duplicates_path):
            with open(self.duplicates_path, "r", encoding="utf-8") as f:
                dup_data = json.load(f)
                for doc_list in dup_data.get("exact", {}).values():
                    if len(doc_list) > 1:
                        self.duplicate_sets.add(tuple(sorted(doc_list)))
    
    
    def _tokenize_query(self, query):
        tokens = []
        for match in TOKEN_PATTERN.finditer(query):
            raw = match.group(0).lower()
            stem = self.stemmer.stem(raw)
            tokens.append(stem)
        return tokens
    
    def _get_postings_from_index(self, term, index_path):
        if not os.path.exists(index_path):
            return [], 0
        
        cache_key = f"{index_path}:{term}"
        if cache_key in self._posting_cache:
            return self._posting_cache[cache_key]
        
        file_size = os.path.getsize(index_path)
        if file_size > 100 * 1024 * 1024:  # 100MB
            result = self._binary_search_index(term, index_path, cache_key)
            if result is not None:
                return result
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        parts = line.split("\t", 1)
                        if len(parts) == 2:
                            if parts[0] == term:
                                data = json.loads(parts[1])
                                result = (data.get("postings", []), data.get("df", 0))
                                self._posting_cache[cache_key] = result
                                if len(self._posting_cache) > 10000:
                                    self._posting_cache.clear()
                                return result
                            elif parts[0] > term:
                                break
                    except Exception:
                        continue
        except Exception:
            pass
        
        self._posting_cache[cache_key] = ([], 0)
        return [], 0
    
    def _binary_search_index(self, term, index_path, cache_key):
        try:
            file_size = os.path.getsize(index_path)
            with open(index_path, "rb") as f:
                left, right = 0, file_size
                iterations = 0
                max_iterations = 100
                
                while left < right and iterations < max_iterations:
                    iterations += 1
                    mid = (left + right) // 2
                    
                    f.seek(mid)
                    if mid > 0:
                        chunk_size = min(1024, mid)
                        f.seek(max(0, mid - chunk_size))
                        chunk = f.read(chunk_size)
                        last_newline = chunk.rfind(b'\n')
                        if last_newline >= 0:
                            f.seek(mid - chunk_size + last_newline + 1)
                        else:
                            f.seek(max(0, mid - chunk_size))
                    
                    line_start = f.tell()
                    line = f.readline()
                    
                    if not line:
                        if mid >= right - 1:
                            break
                        left = mid + 1
                        continue
                    
                    try:
                        line_str = line.decode('utf-8', errors='ignore').strip()
                        if not line_str:
                            left = f.tell()
                            continue
                        
                        parts = line_str.split("\t", 1)
                        if len(parts) < 2:
                            left = f.tell()
                            continue
                        
                        current_term = parts[0]
                        
                        if current_term == term:
                            data = json.loads(parts[1])
                            result = (data.get("postings", []), data.get("df", 0))
                            self._posting_cache[cache_key] = result
                            if len(self._posting_cache) > 10000:
                                self._posting_cache.clear()
                            return result
                        elif current_term < term:
                            left = f.tell()
                        else:
                            right = line_start
                            
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, IndexError):
                        left = f.tell()
                        continue
                
                result = ([], 0)
                self._posting_cache[cache_key] = result
                return result
        except Exception:
            return None
    
    def _compute_tfidf(self, tf, df, doc_length=1):
        if df == 0:
            return 0.0
        idf = math.log(self.total_docs / df)
        normalized_tf = 1 + math.log(tf) if tf > 0 else 0
        return normalized_tf * idf
    
    def _get_ngram_postings(self, query_tokens, n):
        if n == 2:
            index_path = self.final_bigram_index_path
        elif n == 3:
            index_path = self.final_trigram_index_path
        else:
            return {}
        
        ngram_scores = defaultdict(float)
        for i in range(len(query_tokens) - n + 1):
            ngram = "_".join(query_tokens[i:i+n])
            postings, df = self._get_postings_from_index(ngram, index_path)
            if postings:
                for posting in postings:
                    if len(posting) >= 2:
                        doc_id = posting[0]
                        tf = posting[1] if len(posting) > 1 else 1
                        score = self._compute_tfidf(tf, df)
                        ngram_scores[doc_id] += score * 0.3
        return ngram_scores
    
    def search(self, query, top_k=10):
        start_time = time.time()
        
        query_tokens = self._tokenize_query(query)
        if not query_tokens:
            return [], 0
        
        doc_scores = defaultdict(float)
        doc_info = {}
        
        all_postings = {}
        all_dfs = {}
        for term in query_tokens:
            postings, df = self._get_postings_from_index(term, self.final_index_path)
            if postings and df > 0:
                all_postings[term] = postings
                all_dfs[term] = df
        
        for term in query_tokens:
            if term not in all_postings:
                continue
            postings = all_postings[term]
            df = all_dfs[term]
            idf = math.log(self.total_docs / df) if df > 0 else 0
            
            for posting in postings:
                if len(posting) < 2:
                    continue
                doc_id = posting[0]
                tf = posting[1]
                important_tf = posting[2] if len(posting) > 2 else 0
                positions = posting[3] if len(posting) > 3 else []
                
                if doc_id not in doc_info:
                    doc_info[doc_id] = {
                        "tf_sum": 0,
                        "important_tf_sum": 0,
                        "positions": [],
                        "term_scores": []
                    }
                
                normalized_tf = 1 + math.log(tf) if tf > 0 else 0
                term_score = normalized_tf * idf
                doc_info[doc_id]["term_scores"].append(term_score)
                doc_info[doc_id]["tf_sum"] += tf
                doc_info[doc_id]["important_tf_sum"] += important_tf
                if positions:
                    doc_info[doc_id]["positions"].extend(positions)
        
        for doc_id, info in doc_info.items():
            base_score = sum(info["term_scores"])
            important_boost = 1.0 + (info["important_tf_sum"] * 0.5)
            
            position_boost = 1.0
            if info["positions"] and len(query_tokens) > 1:
                positions = sorted(set(info["positions"]))
                min_pos = positions[0]
                consecutive_count = 1
                for i, token in enumerate(query_tokens[1:], 1):
                    if token in all_postings:
                        for p in all_postings[token]:
                            if len(p) > 3 and p[0] == doc_id:
                                token_positions = p[3] if isinstance(p[3], list) else []
                                if token_positions and any(abs(pos - min_pos - i) <= 5 for pos in token_positions):
                                    consecutive_count += 1
                                    position_boost += 0.3
                                    break
                if consecutive_count == len(query_tokens):
                    position_boost += 0.5
            
            doc_scores[doc_id] = base_score * important_boost * position_boost
        
        bigram_scores = self._get_ngram_postings(query_tokens, 2)
        trigram_scores = self._get_ngram_postings(query_tokens, 3)
        
        anchor_scores = defaultdict(float)
        for term in query_tokens:
            postings, df = self._get_postings_from_index(term, self.final_anchor_index_path)
            if postings and df > 0:
                idf = math.log(self.total_docs / df) if df > 0 else 0
                for posting in postings:
                    if len(posting) < 2:
                        continue
                    doc_id = posting[0]
                    tf = posting[1] if len(posting) > 1 else 1
                    normalized_tf = 1 + math.log(tf) if tf > 0 else 0
                    score = normalized_tf * idf * 1.5
                    anchor_scores[doc_id] += score
        
        for doc_id, score in bigram_scores.items():
            doc_scores[doc_id] += score
        
        for doc_id, score in trigram_scores.items():
            doc_scores[doc_id] += score
        
        for doc_id, score in anchor_scores.items():
            doc_scores[doc_id] += score
        
        for doc_id in doc_scores:
            if doc_id in self.pagerank:
                doc_scores[doc_id] *= (1.0 + self.pagerank[doc_id] * 2.5)
            if doc_id in self.authorities:
                doc_scores[doc_id] *= (1.0 + self.authorities[doc_id] * 1.5)
            
            if doc_id in self.docid_to_url:
                url = self.docid_to_url[doc_id]
                url_lower = url.lower()
                error_patterns = [
                    '/error', '/404', '/403', '/forbidden', '/denied',
                    'whoops', 'trouble locating', 'trouble', 'apache/',
                    'not found', 'access denied', 'forbidden', 'page not found',
                    'error page', 'server error', 'internal error'
                ]
                if any(pattern in url_lower for pattern in error_patterns):
                    doc_scores[doc_id] *= 0.001
                
                if any(url_lower.endswith(suffix) for suffix in ['/error', '/404', '/403', '/denied', '/forbidden']):
                    doc_scores[doc_id] *= 0.001
                
                if 'apache' in url_lower and ('server' in url_lower or 'port' in url_lower):
                    doc_scores[doc_id] *= 0.001
                
                raw_source_extensions = [
                    '.ff', '.ssi', '.shtml', '.shtm', '.stm', '.inc',
                    '.phtml', '.php3', '.php4', '.phps', '.cgi', '.pl',
                    '.sh', '.py', '.rb', '.jsp', '.asp', '.aspx'
                ]
                url_path = url_lower.split('?')[0]
                for ext in raw_source_extensions:
                    if url_path.endswith(ext):
                        doc_scores[doc_id] *= 0.001
                        break
        
        for dup_set in self.duplicate_sets:
            if len(dup_set) > 1:
                max_score = max(doc_scores.get(d, 0) for d in dup_set)
                for doc_id in dup_set:
                    if doc_id in doc_scores and doc_scores[doc_id] < max_score * 0.9:
                        doc_scores[doc_id] *= 0.1
        
        def is_error_url(url):
            if not url:
                return True
            invalid_patterns = [
                '/error', '/404', '/403', '/forbidden', '/denied',
                'whoops', 'trouble locating', 'trouble', 'permission to access',
                'apache/', 'server at', 'port 80', 'port 443',
                'not found', 'access denied', 'forbidden', 'page not found',
                'error page', 'server error', 'internal error'
            ]
            url_lower = url.lower()
            for pattern in invalid_patterns:
                if pattern in url_lower:
                    return True
            if any(url_lower.endswith(suffix) for suffix in ['/error', '/404', '/403', '/denied', '/forbidden']):
                return True
            if 'apache' in url_lower and ('server' in url_lower or 'port' in url_lower):
                return True
            
            raw_source_extensions = [
                '.ff', '.ssi', '.shtml', '.shtm', '.stm', '.inc',
                '.phtml', '.php3', '.php4', '.phps', '.cgi', '.pl',
                '.sh', '.py', '.rb', '.jsp', '.asp', '.aspx'
            ]
            url_path = url.lower().split('?')[0]
            for ext in raw_source_extensions:
                if url_path.endswith(ext):
                    return True
            
            if '/~' in url_lower and any(url_lower.endswith(f'/{ext}') for ext in raw_source_extensions):
                return True
            
            return False
        
        def sort_key(item):
            doc_id, score = item
            url = self.docid_to_url.get(doc_id, '')
            is_error = is_error_url(url)
            return (not is_error, score)
        
        sorted_docs = sorted(doc_scores.items(), key=sort_key, reverse=True)
        
        results = []
        seen_urls = set()
        total_count = 0
        for doc_id, score in sorted_docs:
            if doc_id in self.docid_to_url:
                url = self.docid_to_url[doc_id]
                if url not in seen_urls and not is_error_url(url):
                    total_count += 1
                    if len(results) < top_k:
                        results.append((url, score))
                    seen_urls.add(url)
        
        elapsed_time = (time.time() - start_time) * 1000
        return results, elapsed_time, total_count


def main():
    parser = argparse.ArgumentParser(description="Assignment 3 - Search Engine")
    parser.add_argument("--index-dir", required=True, help="Path to index directory")
    parser.add_argument("--query", help="Query string")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()
    
    engine = SearchEngine(args.index_dir)
    
    if args.interactive or not args.query:
        print("Search Engine - Interactive Mode")
        print("Type 'quit' or 'exit' to stop\n")
        while True:
            try:
                query = input("Query: ").strip()
                if not query or query.lower() in ["quit", "exit"]:
                    break
                
                results, elapsed, total_count = engine.search(query, top_k=10)
                print(f"\nFound {total_count} results (showing top {len(results)}, took {elapsed:.2f}ms)\n")
                for i, (url, score) in enumerate(results, 1):
                    print(f"{i}. [{score:.4f}] {url}")
                print()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}\n")
    else:
        results, elapsed, total_count = engine.search(args.query, top_k=10)
        print(f"Query: {args.query}")
        print(f"Found {total_count} results (showing top {len(results)}, took {elapsed:.2f}ms)\n")
        for i, (url, score) in enumerate(results, 1):
            print(f"{i}. [{score:.4f}] {url}")


if __name__ == "__main__":
    main()

