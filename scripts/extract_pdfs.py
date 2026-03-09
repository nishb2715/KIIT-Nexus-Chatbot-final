import pdfplumber
from pathlib import Path
import json, re

# Map each PDF → which folder it goes into
PDF_MAP = {
    "KIIT_STUDENT_HANDBOOK.pdf": ["exams", "university", "campus"],
    "KIITEE_2026.pdf": ["admissions", "fees", "placements", "university"],
    "COURSE_CURRICULUM.pdf": ["curriculum"],
    "ACADEMIC_CALENDAR_1st_2nd.pdf": ["calendar"],
    "ACADEMIC_CALENDAR_3rd_4th.pdf": ["calendar"],
    "ACADEMIC_CALENDAR_5th_8th.pdf": ["calendar"],
    "KIIT_NEXUS.pdf": ["community"],
}

def clean_text(text):
    text = re.sub(r'Student Hand Book.*?Page \d+', '', text)  # remove headers
    text = re.sub(r'KIITEE 2026.*?PROSPECTUS', '', text)       # remove headers
    text = re.sub(r'\n{3,}', '\n\n', text)                      # collapse blank lines
    text = re.sub(r' +', ' ', text)                              # collapse spaces
    return text.strip()

def extract_pdf(pdf_path, output_dir, source_name):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text.append(clean_text(text))

    combined = "\n\n".join(full_text)
    stem = Path(pdf_path).stem

    # Save text file
    txt_path = Path(output_dir) / f"{stem}.txt"
    txt_path.write_text(combined, encoding="utf-8")

    # Save metadata JSON
    meta = {
        "source": source_name,
        "category": Path(output_dir).name,
        "university": "KIIT",
        "last_updated": "2025-2026",
        "verified": True
    }
    (Path(output_dir) / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    print(f"✓ Extracted: {stem} → {output_dir}")

# Run extraction
for pdf_file, categories in PDF_MAP.items():
    pdf_path = f"data/raw/{pdf_file}"
    for cat in categories:
        extract_pdf(pdf_path, f"data/processed/{cat}", pdf_file)

print("\n✅ All PDFs extracted. Now review each .txt file manually!")