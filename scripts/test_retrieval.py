from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

queries = [
    ("What is the fee for CSE B.Tech?",         "fees"),
    ("What is the minimum attendance required?", "exams"),
    ("Who founded KIIT?",                        "university"),
    ("What is KIIT Quest?",                      "community"),
    ("When is the mid semester exam in 2025?",   "calendar"),
    ("What is the tuition cost for computer engineering?", "fees"),
    ("What is the fee of btech cse?",            "fees"),
    ("what are the technical socities at kiit",  "campus"),
    ("what are the tech domains at kiit nexus", "community"),
    ("What are the hostel fees?",               "fees"),
]

print("=== Retrieval Validation ===\n")
all_passed = True

for question, expected_category in queries:
    results      = db.similarity_search(question, k=2)
    top_category = results[0].metadata.get("category", "unknown")
    passed       = "✅" if expected_category in top_category else "❌"
    if "❌" in passed:
        all_passed = False
    print(f"{passed} Q: {question}")
    print(f"   Expected: [{expected_category}] | Got: [{top_category}]")
    print(f"   → {results[0].page_content[:100]}...\n")

if all_passed:
    print("🎉 All checks passed! Move to Phase 3.")
else:
    print("⚠️  Some checks failed. Fix the relevant .txt files and re-run ingest.py")