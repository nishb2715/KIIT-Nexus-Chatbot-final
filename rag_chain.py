import os
import re
import random
from datetime import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

load_dotenv()

CHROMA_DIR = "./chroma_db"

# ---------------------------------------------------------------------------
# 28 randomized greetings for the KIITMate persona
# ---------------------------------------------------------------------------
GREETINGS_POOL = [
    "Hello! I'm KIITMate. How may I assist you today?",
    "Welcome to KIIT! I'm KIITMate, your virtual assistant. What can I help you with?",
    "Good day! KIITMate here. Please let me know how I can support you.",
    "Hi there! I'm KIITMate. How can I make your task easier today?",
    "Greetings! I'm KIITMate, your campus companion. What would you like to know?",
    "Hey! I'm KIITMate. Ready to help you with anything about KIIT!",
    "Welcome! I'm KIITMate. Let me know how I can assist you today.",
    "Hi! I'm KIITMate, here to help you navigate KIIT. What do you need?",
    "Hello and welcome! I'm KIITMate. Ask me anything about KIIT!",
    "Hey there! KIITMate at your service. How can I assist you?",
    "Good to see you! I'm KIITMate — your KIIT guide. What's on your mind?",
    "Hi! KIITMate here, ready to answer your queries about KIIT.",
    "Welcome aboard! I'm KIITMate. What information do you need today?",
    "Hello! KIITMate reporting in! What can I do for you?",
    "Hi! I'm KIITMate. Let's get started — how can I help?",
    "Greetings from KIIT! I'm KIITMate. Feel free to ask me anything.",
    "Hey! Welcome to KIITMate. I'm here to make things simpler for you.",
    "Hello! I'm your KIITMate assistant. How may I help you today?",
    "Hi there! I'm KIITMate — your one-stop guide for all things KIIT.",
    "Welcome! KIITMate here. I'm all ears — what would you like to explore?",
    "Hey! I'm KIITMate. Got a question about KIIT? Fire away!",
    "Hello! I'm KIITMate, your smart assistant for KIIT. What's your query?",
    "Hi! Need help with KIIT-related info? KIITMate is here for you!",
    "Welcome to KIIT! I'm KIITMate. Let's find what you're looking for.",
    "Greetings! KIITMate here — your friendly KIIT assistant. How can I help?",
    "Hey! I'm KIITMate. Let me help you with everything KIIT!",
    "Hello! KIITMate is online and ready. What would you like to know?",
    "Hi! I'm KIITMate. Your KIIT journey starts here — ask away!",
]

# Simple patterns to detect if user input is a greeting
_GREETING_PATTERNS = re.compile(
    r"^\s*("
    r"hi\b|helo+\b|hello+\b|hey\b|hii+\b|hola\b|namaste\b"
    r"|good\s*(morning|afternoon|evening|night|day)\b"
    r"|greetings?\b|howdy\b|sup\b|yo\b|what'?s\s*up\b"
    r")\s*[!?.,'\"]*\s*$",
    re.IGNORECASE,
)

# Patterns to detect a greeting at the START of a longer message
_GREETING_PREFIX = re.compile(
    r"^\s*("
    r"hi\b|helo+\b|hello+\b|hey\b|hii+\b|hola\b|namaste\b"
    r"|good\s*(morning|afternoon|evening|night|day)\b"
    r"|greetings?\b|howdy\b|sup\b|yo\b|what'?s\s*up\b"
    r")[!?,.'\" ]*",
    re.IGNORECASE,
)

# Patterns to detect a goodbye
_GOODBYE_PATTERNS = re.compile(
    r"^\s*("
    r"bye\b|bye\s*bye\b|goodbye\b|see\s*ya\b|cya\b|ttyl\b|talk\s*to\s*you\s*later\b|catch\s*ya\b|farewell\b"
    r")\s*[!?.,'\"]*\s*$",
    re.IGNORECASE,
)


_NEXUS_QUESTION_PATTERNS = re.compile(
    r"\b(kiit\s*nexus|nexus\s*communit|nexus\s*domain|nexus\s*project|nexus\s*event|"
    r"kiitnexus|nexus\s*app|nexus\s*chatbot|nexus\s*platform|nexus\s*team)\b",
    re.IGNORECASE,
)

IRRELEVANT_RESPONSE = (
    "I can only answer questions related to KIIT University and the KIIT Nexus community. "
    "Please ask a relevant question."
)


def _time_of_day() -> str:
    """Return 'morning', 'afternoon', or 'evening' based on the current local time."""
    hour = datetime.now().hour
    if hour < 12:
        return "morning"
    elif hour < 17:
        return "afternoon"
    else:
        return "evening"


def is_greeting(text: str) -> bool:
    """Return True if the entire text is just a greeting (no real query)."""
    return bool(_GREETING_PATTERNS.match(text.strip()))


def has_greeting_prefix(text: str) -> bool:
    """Return True if the text starts with a greeting but also contains a real query."""
    stripped = text.strip()
    if is_greeting(stripped):
        return False  # pure greeting, not a prefix
    return bool(_GREETING_PREFIX.match(stripped))


def is_goodbye(text: str) -> bool:
    """Return True if the entire text is just a goodbye."""
    return bool(_GOODBYE_PATTERNS.match(text.strip()))


def is_nexus_question(text: str) -> bool:
    """Return True only when the user is explicitly asking about KIIT NEXUS."""
    return bool(_NEXUS_QUESTION_PATTERNS.search(text.strip()))


def get_random_greeting() -> str:
    """Pick a random greeting from the pool and adjust it for time of day."""
    greeting = random.choice(GREETINGS_POOL)
    tod = _time_of_day()

    # Replace generic "Hello!" / "Hi!" / "Hey!" etc. at the start with time-aware version
    time_greeting = f"Good {tod}!"
    greeting = re.sub(
        r"^(Hello!|Hi!|Hey!|Hi there!|Hey there!|Good day!|Good to see you!)",
        time_greeting,
        greeting,
        count=1,
    )
    return greeting

# ---------------------------------------------------------------------------
# System prompt
# Rule 12 now ONLY fires when the user explicitly asks about KIIT NEXUS.
# The post-processing in streamlit_app.py provides a secondary safety net.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are KIITMate, a helpful and friendly assistant for the Kalinga Institute of Industrial Technology (KIIT), Bhubaneswar.

You answer questions about:
- University info, founder (Prof. Achyuta Samanta), rankings, and contacts
- Exam rules, grading system, and attendance policy
- Fees (tuition, hostel, etc.), scholarships, and finance
- Admissions and the KIITEE process
- Academic calendar and exam schedules for 2025-26
- Course curriculum (CSE, etc.), credits, and electives
- Campus life, hostels, sports, clubs, and societies (KSAC)
- Placements and internships

STRICT RULES:
1. Use the provided Context and Chat History to answer.
2. For multi-part questions (e.g., fees AND founder), address ALL parts using the context.
3. If the answer is not in the context, say: "I don't have that information right now. Please contact the relevant office at pro@kiit.ac.in or the specific department email."
4. Be concise, friendly, and use bullet points for lists.
5. "it", "they", "its" usually refer to KIIT or the subjects mentioned in Chat History.
6. Do NOT start your answer with any greeting or self-introduction. Just answer the question directly.
7. ALWAYS answer strictly in ENGLISH. The ONLY exception is if the user EXPLICITLY includes the exact words "in hindi" in their question. If they do not explicitly say "in hindi", you MUST answer in English.
8. If asked "who made you", "who created you", "who built you", or similar questions about your creation, clearly state that you were created by KIIT NEXUS (the KIIT NEXUS team/founding members).

FEE RULES (IMPORTANT):
9. When the user asks about "fee", "fees", "cost", "charges", "price", or "how much" for ANY B.Tech programme (e.g., B.Tech CSE, Computer Science, IT), look for the fee table in the context. For example:
   - B.Tech CSE / IT / Computer Science = Rs. 2,20,000 per semester
   - B.Tech ECE / ETC = Rs. 2,00,000 per semester
   - B.Tech Civil / Mechanical / Electrical = Rs. 1,75,000 per semester
10. When asked about hostel fees, answer ONLY from the hostel fee data. Include room types, AC/Non-AC, occupancy, and mess fees.

PLACEMENT RULES (IMPORTANT):
11. When asked about "companies", "recruiters", "placements", or "which companies visit KIIT", list ALL companies mentioned in the context. Include company names like Microsoft, TCS, Oracle, Capgemini, Tech Mahindra, Amazon, Google, Goldman Sachs, etc.
12. Always include placement statistics when available: number of companies (750+), offers (5,585+), highest CTC (51 LPA from Microsoft), and placement rate (94%).
13. Mention school-wise placements when relevant: School of Technology (460+ companies, 3,800+ offers), School of Management (161+ companies), School of Biotechnology (96% placement), etc.

SOCIETIES RULES (IMPORTANT):
14. "Technical societies at KIIT" or "societies at KIIT" or "clubs at KIIT" = answer about KSAC (KIIT Student Activity Centre) societies like KRS, Konnexions, IEEE, GDG, MLSA, CyberVault, AISoC, etc.
15. "Tech domains at KIIT NEXUS" or "KIIT NEXUS domains" = answer about KIIT NEXUS project domains: Web Development, Android Development, Flutter Development, Machine Learning.
16. KIIT NEXUS is a student-built application/chatbot — it is NOT a KIIT society. Do NOT confuse KIIT NEXUS domains with KIIT university societies.

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Make sure to resolve any pronouns (like "it", "him", "her", "they") to the specific entities mentioned in the chat history.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""")

def build_chain():
    # Load embeddings — same model used during ingestion
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Load ChromaDB from disk
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    # Use similarity search with k=10 documents
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 10
        }
    )

    # Groq LLM — free, fast, no credit card
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=1200,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Memory — keeps last 5 exchanges so follow-up questions work
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Custom prompt
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=SYSTEM_PROMPT
    )

    # Build the full RAG chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        verbose=False
    )

    return chain, retriever