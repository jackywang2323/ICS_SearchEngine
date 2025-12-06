import os
import time
import sys
from search import SearchEngine

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_queries(index_dir, query_file="test_queries.txt"):
    engine = SearchEngine(index_dir)
    
    with open(query_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]
    
    results = []
    total_time = 0
    
    print("Testing query performance...")
    print("=" * 60)
    
    for i, query in enumerate(queries, 1):
        start = time.time()
        search_results, elapsed, total_count = engine.search(query, top_k=10)
        total_time += elapsed
        
        status = "[PASS]" if elapsed < 300 else "[FAIL]"
        print(f"{status} Query {i:2d}: {query[:40]:<40} | {total_count:5d} results | {elapsed:6.2f}ms")
        
        results.append({
            "query": query,
            "num_results": total_count,
            "time_ms": elapsed,
            "status": "PASS" if elapsed < 300 else "FAIL"
        })
    
    print("=" * 60)
    print(f"Total queries: {len(queries)}")
    print(f"Total time: {total_time:.2f}ms")
    print(f"Average time: {total_time/len(queries):.2f}ms")
    print(f"Queries under 300ms: {sum(1 for r in results if r['time_ms'] < 300)}/{len(results)}")
    
    return results

if __name__ == "__main__":
    import sys
    index_dir = sys.argv[1] if len(sys.argv) > 1 else "dev_index"
    test_queries(index_dir)

