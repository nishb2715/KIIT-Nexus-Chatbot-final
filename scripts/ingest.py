import os
import shutil
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# Configuration
PROCESSED_DIR = "data/processed"
CHROMA_DIR    = "./chroma_db"
BATCH_SIZE    = 500

def load_and_tag_documents():
    all_chunks = []
    processed_path = Path(PROCESSED_DIR)

    # [FIX] Reduced chunk_size (600→400) and chunk_overlap (200→100)
    # Smaller, denser chunks improve retrieval precision for FAQ-style content
    faq_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", "Q:", "A:", "R."]
    )

    processed_files = set()
    for txt_file in processed_path.rglob("*.txt"):
        if txt_file.name in processed_files:
            print(f"  ⚠ Skipping duplicate file: {txt_file.name}")
            continue
        processed_files.add(txt_file.name)
        # Default category from folder name
        folder_category = txt_file.parent.name

        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            docs   = loader.load()
        except Exception as e:
            print(f"  ✗ Could not load {txt_file.name} → {e}")
            continue

        for doc in docs:
            content = doc.page_content

            # 1. Master FAQ file — split by section and tag each section correctly
            if "faq.txt" in txt_file.name.lower():
                sections = re.split(r'={5,}\nSECTION \d+: (.*?)\n={5,}', content)

                # sections[0] is file header, then (Title, Content) pairs follow
                for i in range(1, len(sections), 2):
                    section_title   = sections[i].lower()
                    section_content = sections[i + 1]

                    # Map section title → correct knowledge category
                    if "fee" in section_title:
                        cat = "fees"
                    elif any(k in section_title for k in ["exam", "attendance", "grading"]):
                        cat = "exams"
                    elif "admission" in section_title:
                        cat = "admissions"
                    elif "calendar" in section_title:
                        cat = "calendar"
                    elif "curriculum" in section_title or "course" in section_title:
                        cat = "curriculum"
                    elif "university" in section_title:
                        cat = "university"
                    elif "campus" in section_title or "student life" in section_title:
                        cat = "campus"
                    # [FIX] Societies / clubs sections belong to campus
                    elif "societ" in section_title or "club" in section_title:
                        cat = "campus"
                    elif "placement" in section_title:
                        cat = "placements"
                    elif "discipline" in section_title or "conduct" in section_title:
                        cat = "exams"
                    elif "compliance" in section_title or "contact" in section_title:
                        cat = "university"
                    else:
                        cat = "community"

                    section_chunks = faq_splitter.split_text(section_content)
                    for chunk in section_chunks:
                        all_chunks.append(Document(
                            page_content=chunk,
                            metadata={
                                "category": cat,
                                "source":   txt_file.name
                            }
                        ))

            # 2. Student Handbook — tag attendance/exam chunks correctly
            elif "handbook" in txt_file.name.lower():
                file_chunks = faq_splitter.split_documents([doc])
                for chunk in file_chunks:
                    chunk_text = chunk.page_content.lower()
                    has_fee_indicator = any(k in chunk_text for k in ["rs.", "per semester", "fee", "/-"])
                    if any(k in chunk_text for k in ["attendance", "r.7", "grading", "r.8", "r.9", "r.10", "supplementary", "grade"]):
                        chunk.metadata["category"] = "exams"
                    elif any(k in chunk_text for k in ["scholarship", "fee", "r.19", "tuition"]):
                        chunk.metadata["category"] = "fees"
                    # Hostel chunk with fee indicators → tag as fees
                    elif "hostel" in chunk_text and has_fee_indicator:
                        chunk.metadata["category"] = "fees"
                    elif any(k in chunk_text for k in [
                        "library", "sports", "hostel", "health", "ksac",
                        "counselling", "placement",
                        # [FIX] societies & clubs content belongs to campus
                        "societ", "club", "krs", "konnexion", "ieee", "gdg",
                        "mlsa", "cybervault", "aisoc", "technical club"
                    ]):
                        chunk.metadata["category"] = "campus"
                    elif any(k in chunk_text for k in ["discipline", "conduct", "r.20", "ragging", "sanction"]):
                        chunk.metadata["category"] = "exams"
                    elif any(k in chunk_text for k in ["registration", "r.6", "curriculum", "r.2", "credit", "minor", "honours"]):
                        chunk.metadata["category"] = "curriculum"
                    elif "founder" in chunk_text or "achyuta samanta" in chunk_text:
                        chunk.metadata["category"] = "university"
                    else:
                        chunk.metadata["category"] = folder_category
                    chunk.metadata["source"] = txt_file.name
                    all_chunks.append(chunk)

            # 3. KIITEE file — split across multiple categories
            elif "kiitee" in txt_file.name.lower():
                file_chunks = faq_splitter.split_documents([doc])
                for chunk in file_chunks:
                    chunk_text = chunk.page_content.lower()
                    if any(k in chunk_text for k in ["fee structure", "per semester", "refund", "scholarship", "tuition"]):
                        chunk.metadata["category"] = "fees"
                    elif any(k in chunk_text for k in ["placement", "recruiter", "lpa", "ctc", "job offer"]):
                        chunk.metadata["category"] = "placements"
                    elif any(k in chunk_text for k in ["admission", "kiitee", "counselling", "swc", "eligib"]):
                        chunk.metadata["category"] = "admissions"
                    elif any(k in chunk_text for k in ["rank", "hostel", "library", "sports", "kims", "campus"]):
                        chunk.metadata["category"] = "campus"
                    else:
                        chunk.metadata["category"] = "university"
                    chunk.metadata["source"] = txt_file.name
                    all_chunks.append(chunk)

            # 4. Academic Calendar files
            elif "academic_calendar" in txt_file.name.lower():
                file_chunks = faq_splitter.split_documents([doc])
                for chunk in file_chunks:
                    chunk.metadata["category"] = "calendar"
                    chunk.metadata["source"]   = txt_file.name
                    all_chunks.append(chunk)

            # 5. Course Curriculum
            elif "curriculum" in txt_file.name.lower():
                file_chunks = faq_splitter.split_documents([doc])
                for chunk in file_chunks:
                    chunk.metadata["category"] = "curriculum"
                    chunk.metadata["source"]   = txt_file.name
                    all_chunks.append(chunk)

            # 6. Hostel Fees — dedicated file
            elif "hostel_fee" in txt_file.name.lower() or "hostel_fees" in txt_file.name.lower():
                file_chunks = faq_splitter.split_documents([doc])
                for chunk in file_chunks:
                    chunk.metadata["category"] = "fees"
                    chunk.metadata["source"]   = txt_file.name
                    all_chunks.append(chunk)

            # 7. Societies — dedicated file
            elif "societ" in txt_file.name.lower():
                file_chunks = faq_splitter.split_documents([doc])
                for chunk in file_chunks:
                    chunk.metadata["category"] = "campus"
                    chunk.metadata["source"]   = txt_file.name
                    all_chunks.append(chunk)

            # 8. KIIT NEXUS and all other community files
            else:
                file_chunks = faq_splitter.split_documents([doc])
                for chunk in file_chunks:
                    chunk_text = chunk.page_content.lower()
                    if "tuition" in chunk_text or "fee" in chunk_text:
                        chunk.metadata["category"] = "fees"
                    elif "founder" in chunk_text or "achyuta samanta" in chunk_text:
                        chunk.metadata["category"] = "university"
                    else:
                        chunk.metadata["category"] = folder_category
                    chunk.metadata["source"]   = txt_file.name
                    all_chunks.append(chunk)

        print(f"  ✓ Processed and Meta-tagged: {txt_file.name}")

    print(f"\n📂 Total chunks created: {len(all_chunks)}")
    return all_chunks


def ingest():
    # Clear old database to prevent metadata contamination
    if os.path.exists(CHROMA_DIR):
        print(f"🗑️  Cleaning old ChromaDB at {CHROMA_DIR}...")
        shutil.rmtree(CHROMA_DIR)

    chunks = load_and_tag_documents()

    print("\n⏳ Initializing Embedding Model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    total = len(chunks)
    print(f"⏳ Embedding {total} chunks into ChromaDB in batches of {BATCH_SIZE}...\n")

    vectorstore = None
    for i in range(0, total, BATCH_SIZE):
        batch     = chunks[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"   📦 Batch {batch_num} → chunks {i + 1} to {min(i + BATCH_SIZE, total)}")

        if vectorstore is None:
            # First batch — creates the ChromaDB on disk
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_DIR
            )
        else:
            # Subsequent batches — adds to existing ChromaDB
            vectorstore.add_documents(batch)

    print(f"\n✅ Ingestion Complete! {total} chunks stored in ChromaDB.")
    print("   Run test_retrieval.py to validate, then streamlit run streamlit_app.py")


if __name__ == "__main__":
    print("=== KIIT NEXUS Chatbot — Smart Ingestor ===\n")
    ingest()
