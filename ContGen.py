# ==============================================================
# 🧠 ContGen — Full Integrated Version
# ==============================================================
# Features:
# ✅ Renamed to ContGen
# ✅ Login dropdown: [Login, Forget...]
# ✅ Auto role detection (Owner/Admin/Creator)
# ✅ Owner shown as "Super Admin" (UI only)
# ✅ Dedicated full User Management page (not sidebar expander)
# ✅ All original features preserved (Ingest, Prompt + Post generation, Quota)
# ==============================================================

import os
import io
import re
import json
import hashlib
from typing import List, Dict, Tuple
from uuid import uuid4
import pandas as pd
import torch
import requests
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel
import pdfplumber
import chromadb
from chromadb.config import Settings
from supabase import create_client
from dotenv import load_dotenv

# ---------------------------
# CONFIG
# ---------------------------
load_dotenv()

APP_NAME = "ContGen"
st.set_page_config(page_title=APP_NAME, page_icon=" ", layout="wide")
st.title("ContGen")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OWNER_EMAIL = os.getenv("OWNER_EMAIL")
OWNER_PASSWORD = os.getenv("OWNER_PASSWORD")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
GEN_MODEL = "deepseek/deepseek-r1:free"

CHROMA_DIR_BASE = "chroma_db_unified"
POSTS_COLL = "posts_examples_unified"
BOOKS_COLL = "books_chunks_unified"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------
# CLIENTS
# ---------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
service_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY) if SUPABASE_SERVICE_ROLE_KEY else None
client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

# ---------------------------
# EMBEDDING MODEL
# ---------------------------
class SimpleEmbedder:
    def __init__(self, model_name=EMBED_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def encode(self, texts: List[str], batch_size: int = 16):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                out_emb = self.model(**enc).last_hidden_state.mean(dim=1)
                out_emb = torch.nn.functional.normalize(out_emb, p=2, dim=1)
                out.extend(out_emb.cpu().numpy().tolist())
        return out

embedder = SimpleEmbedder()

# ---------------------------
# BASIC TEXT UTILS
# ---------------------------
def decode_bytes_with_fallback(b: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8", errors="ignore")

def extract_text_from_pdf_fileobj(file_obj) -> str:
    data = file_obj.read()
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)
    except Exception:
        pass
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception:
        pass
    return decode_bytes_with_fallback(data)

def extract_text_from_txt_fileobj(file_obj) -> str:
    data = file_obj.read()
    return decode_bytes_with_fallback(data) if isinstance(data, bytes) else str(data)

def chunk_text(text: str, max_chars: int = 1200):
    paras = [p.strip() for p in re.split(r"\n{1,}", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) < max_chars:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            chunks.append(cur)
            cur = p
    if cur: chunks.append(cur)
    return chunks
# ==============================================================
# 🧩 PART 2 – Book parsing, Chroma setup, Supabase helpers
# ==============================================================

PART_RE = re.compile(r"^\s*(Part\s+[IVXLC0-9A-Za-z\-]+)\b", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^\s*(Chapter\s+(\d+))\b", re.IGNORECASE)
HEADING_RE = re.compile(
    r"^(?:[A-Z][A-Z\s]{3,}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}|Pitfall\s*\d+|Mistake\s*\d+|Lesson\s*\d+|Case Study.*)$",
    re.IGNORECASE,
)

def extract_topics_from_chapter_text(ch_text: str, max_topics: int = 40):
    lines = [l.strip() for l in ch_text.splitlines() if l.strip()]
    topics = []
    for line in lines:
        if HEADING_RE.match(line):
            h = re.sub(r"[^0-9A-Za-z\s\-\&\:\,\(\)\.]", "", line).strip()
            if h and len(h) > 3:
                topics.append(h)
            if len(topics) >= max_topics:
                break
    return topics or [ch_text[:100]]

def parse_book_hierarchy(text: str) -> Dict:
    lines = text.splitlines()
    current_part, current_chapter, buf = None, None, []
    structure = {"parts": {}, "chapters_no_part": {}}

    def flush():
        nonlocal buf, current_part, current_chapter
        if not current_chapter: return
        chap_text = "\n".join(buf).strip()
        if current_part:
            structure["parts"].setdefault(current_part, {"chapters": {}})["chapters"][current_chapter] = chap_text
        else:
            structure["chapters_no_part"][current_chapter] = chap_text
        buf = []

    for l in lines:
        line = l.strip()
        if not line:
            buf.append(""); continue
        if PART_RE.match(line):
            flush(); current_part = PART_RE.match(line).group(1); current_chapter=None; buf=[]
        elif CHAPTER_RE.match(line):
            flush(); current_chapter = CHAPTER_RE.match(line).group(1); buf=[]
        else:
            buf.append(line)
    flush()
    return structure

# ---------------------------
# Supabase helpers
# ---------------------------
def get_app_user_by_id(uid):
    res = supabase.table("app_users").select("*").eq("id", uid).limit(1).execute()
    return res.data[0] if res and res.data else None

def get_app_user_by_email(email):
    res = supabase.table("app_users").select("*").eq("email", email).limit(1).execute()
    return res.data[0] if res and res.data else None

def upsert_app_user_row_with_id(uid, email, role="creator", gen_limit=100, creator_quota=0, created_by=None):
    supabase.table("app_users").upsert({
        "id": uid, "email": email, "role": role,
        "generation_limit": gen_limit, "generation_used": 0,
        "creator_quota": creator_quota, "created_by": created_by, "is_active": True
    }).execute()

def list_app_users(limit=500):
    res = supabase.table("app_users").select("*").limit(limit).execute()
    return res.data or []

def load_user_books_meta(uid):
    res = supabase.table("books_meta").select("metadata").eq("user_id", uid).limit(1).execute()
    if res and res.data and res.data[0].get("metadata"):
        return res.data[0]["metadata"]
    return {}

def save_user_books_meta(uid, meta):
    supabase.table("books_meta").upsert({"user_id": uid, "metadata": meta}).execute()

# ---------------------------
# Owner auto-creation
# ---------------------------
def ensure_owner_exists():
    if not (OWNER_EMAIL and OWNER_PASSWORD and service_client): return
    existing = supabase.table("app_users").select("*").eq("email", OWNER_EMAIL).limit(1).execute()
    if existing.data:
        return
    headers = {"apikey": SUPABASE_SERVICE_ROLE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}"}
    payload = {"email": OWNER_EMAIL, "password": OWNER_PASSWORD, "email_confirm": True}
    r = requests.post(f"{SUPABASE_URL}/auth/v1/admin/users", json=payload, headers=headers)
    uid = r.json().get("id") if r.status_code in (200,201) else None
    if uid:
        upsert_app_user_row_with_id(uid, OWNER_EMAIL, role="owner", gen_limit=-1, creator_quota=-1)

ensure_owner_exists()

# ---------------------------
# Chroma setup
# ---------------------------
chroma_client = None
posts_collection = None
books_collection = None

def init_user_chroma(uid):
    global chroma_client, posts_collection, books_collection
    db_path = f"{CHROMA_DIR_BASE}_{uid}"
    os.makedirs(db_path, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=db_path, settings=Settings(allow_reset=False))
    posts_collection = chroma_client.get_or_create_collection(POSTS_COLL)
    books_collection = chroma_client.get_or_create_collection(BOOKS_COLL)

def ingest_posts_csv_bytes_for_user(csv_bytes: bytes):
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(io.BytesIO(csv_bytes), encoding=enc, dtype=str).fillna("")
            break
        except Exception:
            df=None
    if df is None: return False,"Decode failed"
    if "id" not in df or "content" not in df: return False,"Missing id/content columns"
    if "profile" not in df: df["profile"]="unknown"
    ids,docs,md = df["id"].astype(str),df["content"].astype(str),[{"profile":p} for p in df["profile"]]
    embs = embedder.encode(docs.tolist())
    posts_collection.add(ids=ids.tolist(), documents=docs.tolist(), embeddings=embs, metadatas=md)
    return True,f"Ingested {len(docs)} posts"

def retrieve_style_examples_for_query(q, n=5):
    q_emb = embedder.encode([q])[0]
    res = posts_collection.query(query_embeddings=[q_emb], n_results=n)
    return res.get("documents", [[]])[0]

def retrieve_book_chunks_for_context(q, n=5):
    q_emb = embedder.encode([q])[0]
    res = books_collection.query(query_embeddings=[q_emb], n_results=n)
    return res.get("documents", [[]])[0]
# ==============================================================
# 🧠 PART 3 – Generation logic (prompts & posts) and quota management
# ==============================================================

# ---------------------------
# Generation system prompts
# ---------------------------
PROMPT_SYSTEM = """
You are a Prompt Architect for a digital marketing strategist AI.
Your task is to take a raw 'idea' or 'topic' and turn it into ONE single rich, detailed prompt
that will guide another AI to write a high-quality LinkedIn thought leadership post.

RULES:
- Add context: what angle to take (mistake, myth, strategy shift, efficiency gain).
- Add structure: request Hook → Insight → Framework → Example → CTA.
- Add audience lens: CMO, CEO, CFO, or Director of Digital.
- Ensure uniqueness: each prompt must stand alone.
- Always tie the solution in the prompt to the 5Ws marketing framework.
- Keep the output under 120 words.
"""

PROMPT_5WS = {
    "Who": "Focus on the audience and stakeholders - who benefits, who should act?",
    "What": "Focus on the offering and value proposition - what is it, what changes?",
    "When": "Focus on timing, seasonality, or milestones - when should one act?",
    "Where": "Focus on channels, markets and places - where should the strategy be applied?",
    "Why": "Focus on rationale, business case, and goals - why this approach matters?",
}

# ---------------------------
# Call generation model (OpenRouter / OpenAI wrapper)
# ---------------------------
def call_generation_system(system_prompt: str, user_text: str) -> str:
    chat = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}],
    )
    return chat.choices[0].message.content

def generate_prompts(idea: str, n: int = 3, use_5ws: bool = True) -> List[Dict]:
    out = []
    if use_5ws and n >= 5:
        for w, focus in PROMPT_5WS.items():
            user = f"Create a single prompt (<=120 words) from idea:\n\n{idea}\n\nFocus on {w}: {focus}"
            res = call_generation_system(PROMPT_SYSTEM, user)
            out.append({"w": w, "prompt": res})
        extras = n - 5
        for i in range(extras):
            lens = ["CMO", "CEO", "Director of Digital", "CFO"][i % 4]
            user = f"Create a single prompt (<=120 words) from idea:\n\n{idea}\n\nAudience lens: {lens}"
            res = call_generation_system(PROMPT_SYSTEM, user)
            out.append({"w": None, "prompt": res})
    else:
        for i in range(n):
            lens = ["CMO", "CEO", "Director of Digital", "CFO"][i % 4]
            user = f"Create a single prompt (<=120 words) from idea:\n\n{idea}\n\nAudience lens: {lens}"
            res = call_generation_system(PROMPT_SYSTEM, user)
            out.append({"w": None, "prompt": res})
    return out

# ---------------------------
# Post generation (uses profile template)
# ---------------------------
PROFILE_TEMPLATE = """
You are writing LinkedIn-style posts in the voice of: {profile}

Tone:
- Clear, structured, useful.
- Use a hook, offer an insight, provide an actionable framework, show a short example, and end with a light CTA.

Style Hints (do not copy verbatim; use as tonal guidance):
{style_examples_block}

Book / Topic Inspiration (use these to root the post in subject-matter context):
{book_idea_block}

Constraints:
- Keep to ~220 words (shorter is fine).
- Use simple sentences and a professional voice.
"""

def generate_post(query: str, profile_name: str, style_examples: List[str], book_contexts: List[str]) -> str:
    style_examples_block = "\n\n".join(f"- {s[:300]}..." for s in style_examples) if style_examples else "None"
    book_block = "\n\n".join(b[:600] + ("..." if len(b) > 600 else "") for b in book_contexts) if book_contexts else "None"
    system = PROFILE_TEMPLATE.format(profile=profile_name, style_examples_block=style_examples_block, book_idea_block=book_block)
    chat = client.chat.completions.create(model=GEN_MODEL, messages=[{"role": "system", "content": system}, {"role": "user", "content": f"Write a LinkedIn post about: {query}"}])
    return chat.choices[0].message.content

# ---------------------------
# Quota helpers: check & consume
# ---------------------------
def update_generation_usage_by_email(email: str, increment: int = 1) -> Tuple[bool, str]:
    """
    Increase generation_used for user and their admin (if any).
    Returns (ok, message). If limit exceeded returns False.
    """
    try:
        user_res = supabase.table("app_users").select("*").eq("email", email).limit(1).execute()
        if not user_res.data:
            return False, f"No app_users row for {email}"
        user = user_res.data[0]
        # owner bypass
        if OWNER_EMAIL and user.get("email") and user.get("email").lower() == OWNER_EMAIL.lower():
            return True, "Owner unlimited"
        limit = user.get("generation_limit", -1)
        used = user.get("generation_used", 0) or 0
        if limit != -1 and used + increment > limit:
            return False, f"Quota exceeded ({used}/{limit})"

        # increment user's generation_used
        new_used = used + increment
        supabase.table("app_users").update({"generation_used": new_used}).eq("email", email).execute()

        # if created_by (admin), increment admin used too
        created_by = user.get("created_by")
        if created_by:
            admin_res = supabase.table("app_users").select("*").eq("email", created_by).limit(1).execute()
            if admin_res.data:
                admin = admin_res.data[0]
                if not (OWNER_EMAIL and admin.get("email") and admin.get("email").lower() == OWNER_EMAIL.lower()):
                    admin_limit = admin.get("generation_limit", -1)
                    admin_used = admin.get("generation_used", 0) or 0
                    if admin_limit != -1 and admin_used + increment > admin_limit:
                        # rollback user increment (best effort)
                        supabase.table("app_users").update({"generation_used": used}).eq("email", email).execute()
                        return False, f"Admin quota insufficient ({admin_used}/{admin_limit})"
                    new_admin_used = admin_used + increment
                    supabase.table("app_users").update({"generation_used": new_admin_used}).eq("email", created_by).execute()

        return True, f"Usage updated ({new_used}/{limit})"
    except Exception as e:
        print("update_generation_usage_by_email error:", e)
        return False, str(e)

def update_generation_usage(user_id: str, increment: int = 1) -> Tuple[bool, str]:
    user = get_app_user_by_id(user_id)
    if not user:
        return False, "User not found."
    return update_generation_usage_by_email(user.get("email"), increment=increment)
# ==============================================================
# 🧠 PART 4 – Auth system, Streamlit session, sidebar, and navigation
# ==============================================================

# ---------------------------
# Streamlit Session State
# ---------------------------
if "user" not in st.session_state:
    st.session_state["user"] = None
if "last_prompts" not in st.session_state:
    st.session_state["last_prompts"] = []
if "last_post" not in st.session_state:
    st.session_state["last_post"] = None
if "last_prompt_clicked" not in st.session_state:
    st.session_state["last_prompt_clicked"] = None

# ---------------------------
# Sidebar Auth UI (Login + Forget)
# ---------------------------
st.sidebar.header("Account")
auth_menu = st.sidebar.selectbox("Action", ["Login", "Forgot Password"])

if not st.session_state["user"]:
    if auth_menu == "Login":
        st.sidebar.subheader("Login to ContGen")
        li_email = st.sidebar.text_input("Email", key="li_email")
        li_pass = st.sidebar.text_input("Password", type="password", key="li_pass")
        if st.sidebar.button("Login"):
            try:
                user_resp = supabase.auth.sign_in_with_password({"email": li_email, "password": li_pass})
                if user_resp and getattr(user_resp, "user", None):
                    user_obj = user_resp.user
                    existing = get_app_user_by_id(user_obj.id)
                    if not existing:
                        role_to_insert = "owner" if (OWNER_EMAIL and user_obj.email and user_obj.email.lower() == OWNER_EMAIL.lower()) else "creator"
                        upsert_app_user_row_with_id(
                            user_obj.id, user_obj.email, role=role_to_insert,
                            gen_limit=-1 if role_to_insert == "owner" else 100,
                            creator_quota=-1 if role_to_insert == "owner" else 0,
                            created_by=None
                        )
                    st.session_state["user"] = {"id": user_obj.id, "email": user_obj.email}
                    st.success(f"Logged in as {user_obj.email}")
                    st.rerun()
                else:
                    st.sidebar.error("Login failed. Check credentials.")
            except Exception as e:
                st.sidebar.error(f"Login error: {e}")

    elif auth_menu == "Forgot Password":
        st.sidebar.subheader("Reset Password")
        fp_email = st.sidebar.text_input("Your account email", key="fp_email")
        if st.sidebar.button("Send reset email"):
            try:
                supabase.auth.reset_password_for_email(fp_email, {"redirect_to": ""})
                st.sidebar.success("Password reset email sent.")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

# If user not logged in, stop here
if not st.session_state["user"]:
    st.info("Please log in to continue.")
    st.stop()

# ---------------------------
# Post-login sidebar header
# ---------------------------
user = st.session_state["user"]
USER_ID = user["id"]
USER_EMAIL = user["email"]

# Initialize Chroma for this user
init_user_chroma(USER_ID)
books_meta = load_user_books_meta(USER_ID) or {}

# Retrieve app user row
appu = get_app_user_by_id(USER_ID)
is_owner = OWNER_EMAIL and USER_EMAIL.lower() == OWNER_EMAIL.lower()
display_role = "Super Admin" if is_owner else (appu.get("role") if appu else "unknown")

st.sidebar.markdown(f"**Logged in as:** {USER_EMAIL}")
if st.sidebar.button("Logout"):
    supabase.auth.sign_out()
    st.session_state["user"] = None
    st.rerun()

st.sidebar.markdown(f"**Role:** {display_role}")

# Show quotas
if appu:
    gen_limit = appu.get("generation_limit", -1)
    gen_used = appu.get("generation_used", 0) or 0
    st.sidebar.markdown(f"**Generations used:** {gen_used} / {('∞' if gen_limit == -1 else gen_limit)}")
    

# ---------------------------
# Navigation Tabs
# ---------------------------
nav_options = ["Prompt Generator", "Post Generator"]
if is_owner or (appu and appu.get("role") == "admin"):
    nav_options.append("User Management")

page = st.sidebar.radio("Go to", nav_options)
# ==============================================================
# 🧠 PART 5 – Full-page User Management, Prompt Generator, Post Generator, Ingest UI
# ==============================================================

# ---------------------------
# Helper: create user via Supabase Admin API (server-side)
# ---------------------------
def create_user_via_admin_api(new_email: str, new_password: str):
    """
    Returns (ok, uid or error_msg)
    Requires SUPABASE_SERVICE_ROLE_KEY to be set.
    """
    if not SUPABASE_SERVICE_ROLE_KEY:
        return False, "SUPABASE_SERVICE_ROLE_KEY not configured"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json"
    }
    post_url = f"{SUPABASE_URL}/auth/v1/admin/users"
    payload = {"email": new_email, "password": new_password, "email_confirm": True}
    try:
        r = requests.post(post_url, json=payload, headers=headers, timeout=20)
        if r.status_code in (200, 201):
            data = r.json()
            return True, data.get("id")
        else:
            # try to GET existing
            get_url = f"{SUPABASE_URL}/auth/v1/admin/users?email={new_email}"
            g = requests.get(get_url, headers=headers, timeout=20)
            if g.status_code == 200:
                d = g.json()
                if isinstance(d, list) and len(d) > 0:
                    return True, d[0].get("id")
            return False, f"Auth error: {r.status_code} {r.text}"
    except Exception as e:
        return False, str(e)

def create_user_auth_and_app_row(new_email: str, new_password: str, role: str, generation_limit: int, creator_quota: int, created_by: str) -> Tuple[bool, str]:
    ok, res = create_user_via_admin_api(new_email, new_password)
    if not ok:
        return False, res
    uid = res
    try:
        supabase.table("app_users").upsert({
            "id": uid,
            "email": new_email,
            "role": role,
            "generation_limit": generation_limit,
            "generation_used": 0,
            "creator_quota": creator_quota,
            "created_by": created_by,
            "is_active": True
        }).execute()
        return True, f"User {new_email} created as {role}"
    except Exception as e:
        return False, f"Created auth but failed app_users insert: {e}"

def create_creator_by_admin(admin_id: str, new_email: str, new_password: str, assign_generation_limit: int) -> Tuple[bool, str]:
    admin = get_app_user_by_id(admin_id)
    if not admin:
        return False, "Admin record not found."
    if admin.get("role") != "admin":
        return False, "Only admins can create creators."
    admin_creator_quota = admin.get("creator_quota", 0)
    if admin_creator_quota != -1 and admin_creator_quota < 1:
        return False, "Admin does not have creator quota left."
    admin_limit = admin.get("generation_limit", 0)
    admin_used = admin.get("generation_used", 0) or 0
    if admin_limit != -1 and assign_generation_limit > (admin_limit - admin_used):
        return False, "Admin does not have enough generation_limit to assign."

    ok, msg = create_user_auth_and_app_row(new_email, new_password, role="creator",
                                           generation_limit=assign_generation_limit, creator_quota=0,
                                           created_by=admin.get("email"))
    if not ok:
        return False, msg

    updates = {}
    if admin_creator_quota != -1:
        updates["creator_quota"] = admin_creator_quota - 1
    if admin_limit != -1:
        updates["generation_limit"] = admin_limit - assign_generation_limit
    if updates:
        try:
            supabase.table("app_users").update(updates).eq("id", admin_id).execute()
        except Exception as e:
            print("Failed to deduct admin quotas:", e)
    return True, f"Creator {new_email} created."

# ---------------------------
# Ingest UI (sidebar) - Books & Posts CSV
# ---------------------------
st.sidebar.header("Data Ingest & Storage")
st.sidebar.markdown("Upload books (PDF/TXT) and posts CSV. Files stored in Supabase bucket `user_uploads`.")

with st.sidebar.expander("📚 Ingest Book (PDF / TXT)", expanded=False):
    uploaded_book = st.file_uploader("Choose a book file", type=["pdf", "txt"], key="book_upload")
    book_title = st.text_input("Optional title (friendly)", value="", key="book_title")
    if uploaded_book is not None:
        if st.button("Ingest Book", key="ingest_book_btn"):
            # ingest logic (uses supabase storage + embedding)
            try:
                uploaded_book.seek(0)
                file_bytes = uploaded_book.read()
                ext = (uploaded_book.name or "").lower().split(".")[-1]
                # upload to storage
                filepath = f"{USER_ID}/{uploaded_book.name}"
                try:
                    supabase.storage.from_("user_uploads").upload(filepath, file_bytes, {"content-type": uploaded_book.type or "application/octet-stream"})
                except Exception:
                    # try upsert by remove then upload
                    try:
                        supabase.storage.from_("user_uploads").remove([filepath])
                    except Exception:
                        pass
                    supabase.storage.from_("user_uploads").upload(filepath, file_bytes, {"content-type": uploaded_book.type or "application/octet-stream"})
                # extract text
                if ext == "pdf":
                    text = extract_text_from_pdf_fileobj(io.BytesIO(file_bytes))
                else:
                    text = extract_text_from_txt_fileobj(io.BytesIO(file_bytes))
                if not text or len(text.strip()) < 50:
                    st.error("Could not extract meaningful text from uploaded file.")
                else:
                    hierarchy = parse_book_hierarchy(text)
                    chunks = chunk_text(text, max_chars=1200)
                    doc_ids, docs, metas = [], [], []
                    for i, c in enumerate(chunks):
                        cid = hashlib.sha1(f"{uploaded_book.name}_{i}_{len(c)}".encode()).hexdigest()
                        doc_ids.append(cid)
                        docs.append(c)
                        metas.append({"book": uploaded_book.name, "chunk_index": i})
                    embs = embedder.encode(docs)
                    # add to user's books_collection
                    try:
                        books_collection.add(ids=doc_ids, documents=docs, embeddings=embs, metadatas=metas)
                    except Exception as e:
                        st.error(f"Chroma add failed: {e}")
                        books_collection = chroma_client.get_or_create_collection(BOOKS_COLL)
                        books_collection.add(ids=doc_ids, documents=docs, embeddings=embs, metadatas=metas)
                    # save metadata to books_meta table
                    try:
                        existing = supabase.table("books_meta").select("*").eq("user_id", USER_ID).limit(1).execute()
                        meta_all = existing.data[0].get("metadata") if existing and existing.data else {}
                        meta_all[uploaded_book.name] = {
                            "title": uploaded_book.name,
                            "n_chunks": len(chunks),
                            "hierarchy": hierarchy
                        }
                        supabase.table("books_meta").upsert({"user_id": USER_ID, "metadata": meta_all}).execute()
                        books_meta = load_user_books_meta(USER_ID)
                        st.success(f"Uploaded and indexed {len(chunks)} chunks successfully.")
                    except Exception as e:
                        st.error(f"Metadata save failed: {e}")
            except Exception as e:
                st.error(f"Ingest failed: {e}")

with st.sidebar.expander("📝 Ingest Posts CSV (id,content,profile)", expanded=False):
    uploaded_posts = st.file_uploader("Choose posts CSV", type=["csv"], key="posts_csv")
    if uploaded_posts is not None:
        if st.button("Ingest Posts CSV", key="ingest_posts_btn"):
            try:
                ok, msg = ingest_posts_csv_bytes_for_user(uploaded_posts.getvalue())
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
            except Exception as e:
                st.error(f"Ingest error: {e}")
    try:
        st.markdown(f"**Posts indexed:** {posts_collection.count()}")
    except Exception:
        st.markdown("**Posts indexed:** 0")

# ---------------------------
# PAGE: USER MANAGEMENT (full page)
# ---------------------------
if page == "User Management":
    st.header("User Management")
    if not (is_owner or (appu and appu.get("role") == "admin")):
        st.warning("Access restricted to Admins and Super Admins only.")
    else:
        if is_owner:
            st.subheader("Super Admin — Manage Users")
            st.markdown("Create Admin or Creator accounts, and edit any user's limits.")
            with st.form("create_user_form"):
                cu_email = st.text_input("New user's email")
                cu_pass = st.text_input("Temporary password", type="password")
                cu_role = st.selectbox("Role", ["admin", "creator"])
                cu_gen_limit = st.number_input("Generation limit (-1 for unlimited)", value=100, step=10)
                cu_creator_quota = st.number_input("Creator quota (admins only)", value=0, step=1)
                submitted = st.form_submit_button("Create User")
                if submitted:
                    ok, msg = create_user_auth_and_app_row(cu_email, cu_pass, cu_role, cu_gen_limit, cu_creator_quota if cu_role=="admin" else 0, created_by=USER_EMAIL)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

            st.markdown("---")
            st.subheader("All users")
            users = list_app_users(500)
            for u in users:
                u_email = u.get("email")
                if OWNER_EMAIL and u_email and u_email.lower() == OWNER_EMAIL.lower():
                    st.markdown(f"**{u_email}** — 🔒 Super Admin")
                    continue
                with st.expander(f"{u_email} — {u.get('role')}"):
                    cols = st.columns([3,2,1])
                    cols[0].write(f"Role: {u.get('role')}")
                    cols[1].write(f"Used: {u.get('generation_used',0)} / {('∞' if u.get('generation_limit',-1)==-1 else u.get('generation_limit'))}")
                    cols[2].write("Active" if u.get("is_active", True) else "Inactive")
                    new_limit = st.number_input(f"New generation limit for {u_email}", value=u.get("generation_limit",100), key=f"ul_{u['id']}")
                    new_active = st.checkbox("Active?", value=u.get("is_active", True), key=f"ua_{u['id']}")
                    if st.button(f"Save {u_email}", key=f"us_{u['id']}"):
                        supabase.table("app_users").update({"generation_limit": new_limit, "is_active": new_active}).eq("id", u['id']).execute()
                        st.success("Updated.")
                        st.rerun()

        else:
            st.subheader("Admin — Manage your Creators")
            st.markdown(f"Your creator quota: {('∞' if appu.get('creator_quota',0)==-1 else appu.get('creator_quota',0))}")
            with st.form("admin_create_creator"):
                ac_email = st.text_input("Creator email")
                ac_pass = st.text_input("Creator temporary password", type="password")
                ac_assign_limit = st.number_input("Assign generation limit", value=100, step=10)
                submitted = st.form_submit_button("Create Creator")
                if submitted:
                    ok, msg = create_creator_by_admin(USER_ID, ac_email, ac_pass, ac_assign_limit)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

            st.markdown("---")
            my_creators = supabase.table("app_users").select("*").eq("created_by", USER_EMAIL).execute().data or []
            for c in my_creators:
                with st.expander(f"{c.get('email')} — used {c.get('generation_used',0)}/{('∞' if c.get('generation_limit',-1)==-1 else c.get('generation_limit'))}"):
                    new_limit = st.number_input("New generation limit", value=c.get("generation_limit",100), key=f"acl_{c['id']}")
                    new_active = st.checkbox("Active?", value=c.get("is_active", True), key=f"aca_{c['id']}")
                    if st.button(f"Save {c.get('email')}", key=f"acs_{c['id']}"):
                        supabase.table("app_users").update({"generation_limit": new_limit, "is_active": new_active}).eq("id", c['id']).execute()
                        st.success("Updated.")
                        st.rerun()

    st.stop()  # stop further page rendering when in user management

# ---------------------------
# PAGE: Prompt Generator
# ---------------------------
if page == "Prompt Generator":
    st.header("Prompt Generator")
    st.markdown("Paste idea text or select a book/chapter to seed prompts. Generating N prompts consumes N quota units.")

    col1, col2 = st.columns([2,1])
    with col1:
        idea_text = st.text_area("Idea / Topic (paste chapter/topic text or type an idea):", height=220)
    with col2:
        book_options = list(books_meta.keys()) if books_meta else []
        selected_book = st.selectbox("Select Book (optional)", options=["None"] + book_options, index=0)
        selected_chapter = None
        selected_topic = None
        if selected_book and selected_book != "None":
            hierarchy = books_meta[selected_book]["hierarchy"]
            chapters_all = []
            parts = list(hierarchy.get("parts", {}).keys())
            for p in parts:
                chapters_all.extend(list(hierarchy["parts"][p].get("chapters", {}).keys()))
            chapters_all.extend(list(hierarchy.get("chapters_no_part", {}).keys()))
            if chapters_all:
                selected_chapter = st.selectbox("Select Chapter", options=["Select Chapter"] + chapters_all)
                if selected_chapter and selected_chapter != "Select Chapter":
                    chap_text = ""
                    for p in parts:
                        if selected_chapter in hierarchy["parts"][p].get("chapters", {}):
                            chap_text = hierarchy["parts"][p]["chapters"][selected_chapter]
                            break
                    if not chap_text:
                        chap_text = hierarchy["chapters_no_part"].get(selected_chapter, "")
                    topics = extract_topics_from_chapter_text(chap_text)
                    selected_topic = st.selectbox("Topic", options=["All Topics"] + topics)

        n_prompts = st.slider("How many prompts", 1, 10, 3)
        ensure_5ws = st.checkbox("Ensure 5Ws coverage when >=5 prompts", True)
        gen_btn = st.button("Generate Prompt(s)", key="gen_prompts_btn")

    seed_text = idea_text.strip()
    if not seed_text and selected_book and selected_book != "None" and selected_chapter and selected_chapter != "Select Chapter":
        hierarchy = books_meta[selected_book]["hierarchy"]
        chapter_text = ""
        if "parts" in hierarchy:
            for p in hierarchy["parts"].keys():
                if selected_chapter in hierarchy["parts"][p].get("chapters", {}):
                    chapter_text = hierarchy["parts"][p]["chapters"][selected_chapter]
                    break
        if not chapter_text:
            chapter_text = hierarchy["chapters_no_part"].get(selected_chapter, "")
        if selected_topic and selected_topic != "All Topics":
            idx = chapter_text.find(selected_topic)
            seed_text = chapter_text[idx: idx+1200] if idx!=-1 else chapter_text[:1200]
        else:
            seed_text = chapter_text[:2000]

    if gen_btn:
        if not seed_text:
            st.error("Provide idea text or select book/chapter.")
        else:
            ok, msg = update_generation_usage(USER_ID, increment=n_prompts)
            if not ok:
                st.error(msg)
            else:
                with st.spinner("Generating prompts..."):
                    prompts_out = generate_prompts(seed_text, n=n_prompts, use_5ws=ensure_5ws)
                st.success("Prompts generated!")
                st.session_state["last_prompts"] = prompts_out
                st.session_state["last_post"] = None
                st.session_state["last_prompt_clicked"] = None
                st.rerun()

    # Display prompts and CTA to generate posts
    if st.session_state.get("last_prompts"):
        st.markdown("### 🧠 Your Generated Prompts")
        for i, p in enumerate(st.session_state["last_prompts"]):
            label = f"[{p['w']}]" if p["w"] else f"Prompt {i+1}"
            st.subheader(label)
            st.code(p["prompt"])
            if st.button("🪄 Generate Post from this Prompt", key=f"cta_gen_post_{i}"):
                ok2, msg2 = update_generation_usage(USER_ID, increment=1)
                if not ok2:
                    st.error(msg2)
                else:
                    with st.spinner("Generating post..."):
                        style_examples = retrieve_style_examples_for_query(p["prompt"], n=3)
                        book_context_texts = retrieve_book_chunks_for_context(p["prompt"], n=3)
                        post_text = generate_post(p["prompt"], "Default", style_examples, book_context_texts)
                    st.session_state["last_post"] = post_text
                    st.session_state["last_prompt_clicked"] = i
                    st.rerun()

    if st.session_state.get("last_post") and st.session_state.get("last_prompt_clicked") is not None:
        idx = st.session_state["last_prompt_clicked"]
        p = st.session_state["last_prompts"][idx]
        st.success(f"✨ Post generated from prompt: {p['prompt'][:120]}...")
        st.write(st.session_state["last_post"])
        st.download_button("Download Post (.txt)", st.session_state["last_post"], file_name="generated_post.txt")
# ---------------------------
# Helper: Detect profiles from posts collection
# ---------------------------
def get_detected_profiles():
    """
    Returns a list of unique profile names from the posts_collection metadata.
    If none are found, returns ['Default'].
    """
    try:
        if not posts_collection:
            return ["Default"]
        all_metadatas = posts_collection.get()["metadatas"]
        profiles = set()
        for md in all_metadatas:
            if isinstance(md, dict) and md.get("profile"):
                profiles.add(md["profile"])
        return sorted(profiles) if profiles else ["Default"]
    except Exception as e:
        print("get_detected_profiles error:", e)
        return ["Default"]

# ---------------------------
# PAGE: Post Generator
# ---------------------------
if page == "Post Generator":
    st.header("Post Generator")
    st.markdown("Generate a post from your topic/prompt. Quotas apply.")
    q_col, opt_col = st.columns([2,1])

    prefill = st.session_state.get("last_post") or ""
    with q_col:
        post_query = st.text_input("Post topic / headline:", value=prefill)
        k_examples = st.slider("Number of style examples to use", 1, 10, 5)
    with opt_col:
        profiles = retrieve_style_examples_for_query("profile detection example", n=5) or ["Default"]
        # Use detected profiles if posts CSV ingested; fallback if none
        detected_profiles = get_detected_profiles() if posts_collection else ["Default"]
        profile_sel = st.selectbox("Profile voice (from CSV)", options=["Auto"] + detected_profiles + ["Custom"])
        custom_profile = ""
        if profile_sel == "Custom":
            custom_profile = st.text_input("Custom profile name (tone):", value="Freelance Marketer")
        book_opt = st.selectbox("Use Book Context (optional)", options=["None"] + (list(books_meta.keys()) if books_meta else []))
        book_context_texts = []
        if book_opt and book_opt != "None":
            hierarchy = books_meta[book_opt]["hierarchy"]
            chs = []
            for p in hierarchy.get("parts", {}):
                chs.extend(list(hierarchy["parts"][p].get("chapters", {}).keys()))
            chs.extend(list(hierarchy.get("chapters_no_part", {}).keys()))
            chapter_selected = st.selectbox("Chapter (optional)", options=["Select Chapter"] + chs)
            if chapter_selected and chapter_selected != "Select Chapter":
                chap_text = ""
                for p in hierarchy.get("parts", {}):
                    if chapter_selected in hierarchy["parts"][p].get("chapters", {}):
                        chap_text = hierarchy["parts"][p]["chapters"][chapter_selected]
                        break
                if not chap_text:
                    chap_text = hierarchy["chapters_no_part"].get(chapter_selected, "")
                book_context_texts = [chap_text]

        temp = st.slider("Creativity (temperature proxy)", 0.0, 1.0, 0.7)
        gen_post_btn = st.button("Generate Post", key="gen_post_btn")

    if gen_post_btn:
        if not post_query.strip():
            st.error("Please enter a post topic / headline.")
        else:
            ok, msg = update_generation_usage(USER_ID, increment=1)
            if not ok:
                st.error(msg)
            else:
                profile_name = custom_profile if profile_sel == "Custom" else (profile_sel if profile_sel != "Auto" else (detected_profiles[0] if detected_profiles else "Default"))
                with st.spinner("Generating post..."):
                    style_examples = retrieve_style_examples_for_query(post_query, n=k_examples)
                    if not book_context_texts and book_opt and book_opt != "None":
                        book_context_texts = retrieve_book_chunks_for_context(post_query, n=3)
                    generated = generate_post(post_query, profile_name, style_examples, book_context_texts)
                st.subheader("Generated Post")
                st.write(generated)
                if style_examples:
                    with st.expander("Style examples used"):
                        for s in style_examples:
                            st.markdown(f"- {s[:400]}...")
                if book_context_texts:
                    with st.expander("Book contexts used"):
                        for b in book_context_texts:
                            st.markdown(f"- {b[:400]}...")
                st.download_button("Download post (.txt)", generated, file_name="generated_post.txt")
                # clear any leftover session prompts/post
                st.session_state["last_post"] = None
                st.session_state["last_prompt_clicked"] = None

# ---------------------------
# Footer / Notes
# ---------------------------
st.markdown("---")
st.markdown("")
