from flask import Flask, render_template, request, jsonify
from search import SearchEngine
import os

app = Flask(__name__)

INDEX_DIR = os.path.join(os.path.dirname(__file__), "dev_index")
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = SearchEngine(INDEX_DIR)
    return engine

@app.route("/")
def index():
    return render_template("search.html")

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"results": [], "time": 0, "count": 0})
    
    search_engine = get_engine()
    results, elapsed, total_count = search_engine.search(query, top_k=20)
    
    formatted_results = [
        {"url": url, "score": round(score, 4)} 
        for url, score in results
    ]
    
    return jsonify({
        "results": formatted_results,
        "time": round(elapsed, 2),
        "count": len(formatted_results),
        "total_count": total_count,
        "query": query
    })

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)

