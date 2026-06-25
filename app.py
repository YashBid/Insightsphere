"""
InsightSphere — AI-Powered Financial Research Assistant
=======================================================
A Streamlit application that uses Retrieval-Augmented Generation (RAG) to
analyze financial news articles and answer user questions with source-grounded
AI responses.

Stack:
    - Streamlit (UI)
    - LangChain + FAISS (RAG pipeline)
    - HuggingFace Sentence Transformers (embeddings)
    - Groq API (LLM)
    - SQLite + bcrypt (user authentication)

Usage:
    streamlit run app.py
"""

import os
import pickle
import re
import sqlite3
import time
from typing import Optional

import bcrypt
import html2text
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from readability import Document as ReadabilityDocument

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="InsightSphere",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="collapsed",
)

# ─── Environment Setup ────────────────────────────────────────────────────────

load_dotenv()

groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found! Please set it in the .env file.")
    st.stop()

# ─── Constants ────────────────────────────────────────────────────────────────

DB_PATH = "/tmp/insightsphere.db" if os.path.exists("/tmp") else "insightsphere.db"
VECTOR_STORE_PATH = "faiss_store_groq.pkl"
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq — active production model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_URLS = 3
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_CONTENT_LENGTH = 300
MIN_CHUNK_LENGTH = 100

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        /* ── Design Tokens ───────────────────────────── */
        :root {
            --bg-base:      #07090f;
            --bg-surface:   #0d1117;
            --bg-glass:     rgba(255,255,255,0.04);
            --glass-border: rgba(255,255,255,0.08);
            --primary:      #4f8ef7;
            --primary-glow: rgba(79,142,247,0.35);
            --accent:       #818cf8;
            --accent-2:     #a78bfa;
            --success:      #34d399;
            --warning:      #fbbf24;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted:   #475569;
            --neon-blue:    #38bdf8;
        }

        /* ── Base Reset ───────────────────────────────── */
        html, body, [class*="css"], .stApp {
            font-family: 'Inter', sans-serif !important;
            background-color: var(--bg-base) !important;
            color: var(--text-primary) !important;
        }
        .stApp {
            background: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(79,142,247,0.12) 0%, transparent 70%),
                        radial-gradient(ellipse 60% 40% at 90% 80%, rgba(167,139,250,0.08) 0%, transparent 60%),
                        var(--bg-base) !important;
            min-height: 100vh;
        }
        .block-container {
            padding-top: 2rem !important;
            max-width: 900px;
        }

        /* ── Streamlit Overrides ─────────────────────── */
        .stMarkdown, .stText, p, li, span, label {
            color: var(--text-primary) !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
        }
        .stTextInput > label, .stTextInput label {
            color: var(--text-secondary) !important;
            font-size: 0.8rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .stTextInput > div > div > input {
            background: rgba(255,255,255,0.05) !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: 10px !important;
            color: var(--text-primary) !important;
            padding: 12px 14px !important;
            font-size: 0.95rem !important;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .stTextInput > div > div > input:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px var(--primary-glow) !important;
            outline: none !important;
        }
        .stTextInput > div > div > input::placeholder {
            color: var(--text-muted) !important;
        }

        /* ── Buttons ─────────────────────────────────── */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.65rem 1.4rem !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            letter-spacing: 0.02em;
            transition: all 0.25s ease !important;
            box-shadow: 0 4px 15px var(--primary-glow) !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px var(--primary-glow) !important;
            filter: brightness(1.1);
        }
        .stButton > button:active {
            transform: translateY(0px) !important;
        }

        /* ── Sidebar ─────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: rgba(13,17,23,0.95) !important;
            border-right: 1px solid var(--glass-border) !important;
            backdrop-filter: blur(20px);
        }
        [data-testid="stSidebar"] * {
            color: var(--text-primary) !important;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] .stMarkdown p {
            color: var(--text-primary) !important;
        }
        [data-testid="stSidebar"] .stTextInput > div > div > input {
            background: rgba(255,255,255,0.06) !important;
            border: 1px solid var(--glass-border) !important;
            color: var(--text-primary) !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: var(--glass-border) !important;
        }
        .sidebar-label {
            color: var(--text-secondary) !important;
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.8rem;
            display: block;
        }
        .sidebar-info {
            background: rgba(79,142,247,0.08);
            border: 1px solid rgba(79,142,247,0.2);
            border-radius: 8px;
            padding: 0.7rem 0.9rem;
            font-size: 0.82rem;
            color: var(--text-secondary) !important;
            margin-bottom: 1rem;
        }

        /* ── Alerts / Status ─────────────────────────── */
        .stSuccess > div, .stSuccess {
            background: rgba(52,211,153,0.08) !important;
            border: 1px solid rgba(52,211,153,0.25) !important;
            color: var(--success) !important;
            border-radius: 10px !important;
        }
        .stError > div, .stError {
            background: rgba(248,113,113,0.08) !important;
            border: 1px solid rgba(248,113,113,0.25) !important;
            border-radius: 10px !important;
        }
        .stWarning > div, .stWarning {
            background: rgba(251,191,36,0.08) !important;
            border: 1px solid rgba(251,191,36,0.25) !important;
            border-radius: 10px !important;
        }
        .stSpinner > div {
            border-color: var(--primary) transparent transparent transparent !important;
        }

        /* ── Progress Bar ────────────────────────────── */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, var(--primary), var(--accent-2)) !important;
            border-radius: 4px !important;
        }
        .stProgress > div > div > div {
            background: rgba(255,255,255,0.06) !important;
            border-radius: 4px !important;
        }

        /* ── Divider ─────────────────────────────────── */
        hr {
            border: none !important;
            border-top: 1px solid var(--glass-border) !important;
            margin: 1.8rem 0 !important;
        }

        /* ── Custom Components ───────────────────────── */
        .hero-section {
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg,
                rgba(79,142,247,0.15) 0%,
                rgba(129,140,248,0.10) 50%,
                rgba(167,139,250,0.08) 100%);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 2.5rem 2rem;
            text-align: center;
            margin-bottom: 2rem;
            backdrop-filter: blur(10px);
            animation: fadeSlideUp 0.7s ease-out both;
        }
        .hero-section::before {
            content: '';
            position: absolute;
            top: -60px; left: -60px;
            width: 200px; height: 200px;
            background: radial-gradient(circle, rgba(79,142,247,0.25), transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }
        .hero-section::after {
            content: '';
            position: absolute;
            bottom: -40px; right: -40px;
            width: 160px; height: 160px;
            background: radial-gradient(circle, rgba(167,139,250,0.2), transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }
        .hero-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #f1f5f9 0%, var(--neon-blue) 50%, var(--accent-2) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0 0 0.4rem 0;
            line-height: 1.1;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: var(--text-secondary);
            font-weight: 400;
            margin: 0;
            letter-spacing: 0.05em;
        }
        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(79,142,247,0.12);
            border: 1px solid rgba(79,142,247,0.3);
            border-radius: 20px;
            padding: 0.3rem 0.8rem;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--primary) !important;
            margin-bottom: 1.2rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .features-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.85rem;
            margin-top: 1.5rem;
        }
        .feature-item {
            background: var(--bg-glass);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            transition: border-color 0.2s, transform 0.2s;
            animation: fadeSlideUp 0.6s ease-out both;
        }
        .feature-item:hover {
            border-color: rgba(79,142,247,0.4);
            transform: translateY(-2px);
        }
        .feature-icon {
            font-size: 1.4rem;
            line-height: 1;
            flex-shrink: 0;
            margin-top: 2px;
        }
        .feature-text {
            font-size: 0.87rem;
            color: var(--text-secondary) !important;
            line-height: 1.5;
        }
        .feature-text strong {
            color: var(--text-primary) !important;
            display: block;
            font-size: 0.9rem;
            margin-bottom: 0.15rem;
        }
        .section-heading {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-primary) !important;
            margin: 0 0 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .section-heading::after {
            content: '';
            flex: 1;
            height: 1px;
            background: var(--glass-border);
        }
        .answer-box {
            background: rgba(79,142,247,0.06);
            border: 1px solid rgba(79,142,247,0.2);
            border-left: 3px solid var(--primary);
            padding: 1.5rem 1.6rem;
            border-radius: 0 12px 12px 0;
            margin: 1.2rem 0;
            color: var(--text-primary) !important;
            line-height: 1.75;
            animation: fadeSlideUp 0.5s ease-out both;
        }
        .answer-box p, .answer-box li,
        .answer-box ul, .answer-box ol,
        .answer-box strong, .answer-box b {
            color: var(--text-primary) !important;
        }
        .answer-box li { margin-bottom: 0.4rem; }
        .answer-box strong, .answer-box b {
            color: var(--neon-blue) !important;
        }
        .source-badge {
            display: inline-flex;
            align-items: center;
            background: rgba(129,140,248,0.12);
            border: 1px solid rgba(129,140,248,0.25);
            color: var(--accent) !important;
            padding: 0.35rem 0.8rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
            margin: 0.25rem 0.3rem 0.25rem 0;
            word-break: break-all;
            transition: background 0.2s;
        }
        .source-badge:hover {
            background: rgba(129,140,248,0.2);
        }
        .timing-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(52,211,153,0.08);
            border: 1px solid rgba(52,211,153,0.2);
            color: var(--success) !important;
            padding: 0.25rem 0.7rem;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        /* ── Siri Orb Logo ───────────────────────────── */
        .siri-orb {
            display: inline-block;
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: conic-gradient(
                from 0deg,
                #4f8ef7 0%,
                #818cf8 20%,
                #a78bfa 35%,
                #38bdf8 50%,
                #34d399 65%,
                #4f8ef7 80%,
                #818cf8 100%
            );
            filter: blur(1px) brightness(1.2);
            animation: siri-spin 3s linear infinite, siri-pulse 2.5s ease-in-out infinite;
            position: relative;
            flex-shrink: 0;
        }
        .siri-orb::after {
            content: '';
            position: absolute;
            inset: 4px;
            border-radius: 50%;
            background: radial-gradient(circle at 35% 30%,
                rgba(255,255,255,0.55) 0%,
                rgba(255,255,255,0.05) 60%,
                transparent 100%);
        }
        .siri-orb-sm {
            width: 28px;
            height: 28px;
        }
        .siri-orb-sm::after { inset: 3px; }
        .siri-orb-lg {
            width: 64px;
            height: 64px;
        }
        .siri-orb-lg::after { inset: 6px; }
        @keyframes siri-spin {
            0%   { filter: blur(1px) brightness(1.15) hue-rotate(0deg); }
            50%  { filter: blur(1.5px) brightness(1.3) hue-rotate(180deg); }
            100% { filter: blur(1px) brightness(1.15) hue-rotate(360deg); }
        }
        @keyframes siri-pulse {
            0%, 100% { transform: scale(1);    box-shadow: 0 0 18px rgba(79,142,247,0.5), 0 0 40px rgba(167,139,250,0.3); }
            50%       { transform: scale(1.08); box-shadow: 0 0 28px rgba(79,142,247,0.75), 0 0 60px rgba(167,139,250,0.5); }
        }
        .logo-lockup {
            display: flex;
            align-items: center;
            gap: 0.7rem;
        }
        .logo-text-wrap {
            display: flex;
            flex-direction: column;
            line-height: 1.2;
        }
        .logo-name {
            font-size: 1.15rem;
            font-weight: 800;
            color: #f1f5f9 !important;
            letter-spacing: -0.01em;
        }
        .logo-tagline {
            font-size: 0.65rem;
            color: #475569 !important;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            font-weight: 500;
        }
        .logo-lockup-center {
            flex-direction: column;
            align-items: center;
            gap: 0.9rem;
        }
        .logo-lockup-center .logo-text-wrap {
            align-items: center;
        }
        @keyframes fadeSlideUp {
            from { opacity: 0; transform: translateY(16px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 12px var(--primary-glow); }
            50%       { box-shadow: 0 0 28px var(--primary-glow); }
        }
        .fade-in {
            animation: fadeSlideUp 0.7s ease-out both;
        }
    </style>
""", unsafe_allow_html=True)


# ─── Database Functions ───────────────────────────────────────────────────────

def create_database() -> None:
    """Initialize the SQLite database and create the users table if needed."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def hash_password(password: str) -> bytes:
    """Return a bcrypt hash of the given plaintext password.

    Args:
        password: The plaintext password to hash.

    Returns:
        A bcrypt-hashed bytes object.
    """
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())


def verify_password(password: str, password_hash: bytes) -> bool:
    """Check whether a plaintext password matches a stored bcrypt hash.

    Args:
        password: The plaintext password to verify.
        password_hash: The stored bcrypt hash to check against.

    Returns:
        True if the password matches, False otherwise.
    """
    return bcrypt.checkpw(password.encode("utf-8"), password_hash)


def register_user(username: str, password: str) -> bool:
    """Register a new user in the database.

    Args:
        username: The desired username (must be unique).
        password: The plaintext password to hash and store.

    Returns:
        True if registration succeeded, False if the username already exists.
    """
    password_hash = hash_password(password)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> bool:
    """Authenticate a user against stored credentials.

    Args:
        username: The username to look up.
        password: The plaintext password to verify.

    Returns:
        True if credentials are valid, False otherwise.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return verify_password(password, result[0])
    return False


# ─── User Interface Pages ─────────────────────────────────────────────────────

def login_page() -> None:
    """Render the login page and handle login form submission."""
    # Centered layout
    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        st.markdown("""
            <div style="text-align:center;padding:2.5rem 0 1.5rem;animation:fadeSlideUp 0.6s ease-out both;">
                <div class="logo-name" style="font-size:2rem;margin-bottom:0.3rem;">InsightSphere</div>
                <div class="logo-tagline" style="font-size:0.8rem;">Financial Research AI</div>
                <div style="font-size:1rem;color:#94a3b8;margin-top:0.8rem;">Sign in to your account</div>
            </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input(
                "Password", type="password", placeholder="Enter your password"
            )
            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
            if st.form_submit_button("Sign In →", use_container_width=True):
                if not username or not password:
                    st.error("Please fill in all fields")
                elif authenticate_user(username, password):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        st.markdown(
            "<div style='text-align:center;margin-top:1.2rem;color:#475569;font-size:0.85rem;'>New here?</div>",
            unsafe_allow_html=True,
        )
        if st.button("Create an account →", use_container_width=True):
            st.session_state["show_register"] = True
            st.rerun()


def register_page() -> None:
    """Render the registration page and handle registration form submission."""
    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        st.markdown("""
            <div style="text-align:center;padding:2.5rem 0 1.5rem;animation:fadeSlideUp 0.6s ease-out both;">
                <div class="logo-name" style="font-size:2rem;margin-bottom:0.3rem;">InsightSphere</div>
                <div class="logo-tagline" style="font-size:0.8rem;">Financial Research AI</div>
                <div style="font-size:1rem;color:#94a3b8;margin-top:0.8rem;">Create your free account</div>
            </div>
        """, unsafe_allow_html=True)

        with st.form("register_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Choose a username")
            password = st.text_input(
                "Password", type="password", placeholder="Create a password (min 6 chars)"
            )
            confirm_password = st.text_input(
                "Confirm Password", type="password", placeholder="Re-enter your password"
            )
            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
            if st.form_submit_button("Create Account →", use_container_width=True):
                if not username or not password or not confirm_password:
                    st.error("Please fill in all fields")
                elif password != confirm_password:
                    st.error("Passwords don't match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                elif register_user(username, password):
                    st.success("✅ Account created! Please sign in.")
                    st.session_state["show_register"] = False
                    st.rerun()
                else:
                    st.error("Username already taken — try another")

        st.markdown(
            "<div style='text-align:center;margin-top:1.2rem;color:#475569;font-size:0.85rem;'>Already have an account?</div>",
            unsafe_allow_html=True,
        )
        if st.button("Sign in instead →", use_container_width=True):
            st.session_state["show_register"] = False
            st.rerun()


# ─── Article Processing Functions ─────────────────────────────────────────────

def call_llm(prompt: str) -> Optional[str]:
    """Send a prompt to the Groq API and return the LLM response.

    Args:
        prompt: The prompt string to send to the model.

    Returns:
        The model's response text, or None if the request failed.
    """
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,  # Lower temperature for more factual responses
    }
    try:
        response = requests.post(
            GROQ_URL, headers=headers, json=data, timeout=30
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            st.error(f"Groq API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"API request failed: {str(e)}")
        return None


def extract_main_content(html_content: str) -> str:
    """Extract the main article body from raw HTML using readability-lxml.

    Args:
        html_content: Raw HTML string from the fetched page.

    Returns:
        HTML string containing only the main article content.
    """
    try:
        doc = ReadabilityDocument(html_content)
        return doc.summary()
    except Exception as e:
        st.warning(f"Content extraction warning: {str(e)}")
        return html_content


def clean_text(text: str) -> str:
    """Remove noise patterns (emails, phone numbers, tracking phrases) from text.

    Args:
        text: Raw extracted text to clean.

    Returns:
        Cleaned text with noise patterns removed and whitespace normalized.
    """
    text = re.sub(r"\S+@\S+", "", text)                                    # Remove emails
    text = re.sub(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]", "", text)       # Remove phone numbers
    text = re.sub(r"Read\s+more\s+at\s+.+", "", text, flags=re.IGNORECASE)  # Remove "Read more at..."
    text = re.sub(r"\s+", " ", text).strip()                               # Normalize whitespace
    return text


def fetch_and_summarize_data_from_urls(urls: list[str]) -> list[Document]:
    """Fetch articles from URLs, extract content, and generate AI summaries.

    For each URL this function:
        1. Fetches the HTML content
        2. Extracts the main article body via readability
        3. Converts HTML to clean plain text
        4. Stores the full text as a LangChain Document
        5. Generates a structured financial summary via OpenRouter

    Args:
        urls: List of article URLs to process.

    Returns:
        List of LangChain Document objects (full text + summaries).
    """
    documents: list[Document] = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, url in enumerate(urls):
        if not url or not url.startswith("http"):
            continue

        try:
            status_text.text(f"Processing URL {i+1}/{len(urls)}: {url[:50]}...")
            progress_bar.progress((i + 0.2) / len(urls))

            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            # Extract main content and convert to plain text
            main_content = extract_main_content(response.text)
            h = html2text.HTML2Text()
            h.ignore_links = True
            h.ignore_images = True
            h.ignore_emphasis = True
            text_content = h.handle(main_content)
            text_content = clean_text(text_content)

            if len(text_content) < MIN_CONTENT_LENGTH:
                st.warning(f"Content too short from URL: {url}")
                continue

            # Store full text document
            documents.append(
                Document(
                    page_content=text_content,
                    metadata={"source": url, "type": "full_text"},
                )
            )

            progress_bar.progress((i + 0.5) / len(urls))

            # Build summarization prompt
            # Note: only the first 8000 chars of the article are sent to stay within token limits
            article_excerpt = text_content[:8000]
            summary_prompt = f"""Extract key financial information from this article in bullet points:
{article_excerpt}

Include:
- Company names mentioned
- Financial figures (revenue, profit, growth rates, etc.)
- Important events or announcements
- Analyst opinions or forecasts
- Market impacts or implications

Exclude:
- Advertisement content
- Marketing language
- Generic financial advice
- Subscription prompts

Format the response as clean, concise bullet points without source references.
Use professional financial terminology.
"""

            summary = call_llm(summary_prompt)
            progress_bar.progress((i + 0.8) / len(urls))

            if summary:
                documents.append(
                    Document(
                        page_content=summary,
                        metadata={"source": url, "type": "summary"},
                    )
                )

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch {url}: {str(e)}")
        except Exception as e:
            st.error(f"Error processing {url}: {str(e)}")

    progress_bar.progress(1.0)
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    return documents


def answer_question(query: str, vectorstore: FAISS) -> tuple[str, set]:
    """Answer a question using the FAISS vector store and OpenRouter LLM.

    Performs a similarity search to find the most relevant document chunks,
    filters out low-quality content, constructs a context-grounded prompt,
    and returns the AI-generated answer along with source URLs.

    Args:
        query: The user's natural language question.
        vectorstore: The FAISS vector store populated with article chunks.

    Returns:
        A tuple of (answer_text, set_of_source_urls).
    """
    relevant_docs = vectorstore.similarity_search(query, k=5)

    # Filter out low-quality chunks (ads, navigation, etc.)
    filtered_docs = [
        doc
        for doc in relevant_docs
        if len(doc.page_content) >= 50
        and "subscribe" not in doc.page_content.lower()
        and "advertisement" not in doc.page_content.lower()
    ]

    if not filtered_docs:
        return "I couldn't find relevant information in the provided articles.", set()

    # Build context from the top 4 most relevant chunks
    context = ""
    sources: set = set()
    for i, doc in enumerate(filtered_docs[:4]):
        context += f"DOCUMENT EXTRACT {i+1}:\n{doc.page_content}\n\n"
        sources.add(doc.metadata["source"])

    prompt = f"""You are a professional financial research assistant. Answer the question using ONLY the provided context.
If you don't know the answer, say "I couldn't find that information in the provided articles."

Context:
{context}

Question: {query}

Guidelines for your response:
1. Be precise and factual - only use information from the context
2. Never mention sources in your answer
3. If different documents contradict, mention this without citing sources
4. Don't make up information not present in the context
5. Format your answer professionally:
   - Use bullet points for lists
   - Bold important figures
   - Separate different points clearly
6. Exclude any marketing or promotional content
7. Focus on financial data, company performance, and market impacts

Answer:"""

    response = call_llm(prompt)
    if not response:
        response = "I couldn't generate a response at this time. Please try again later."

    return response, sources


# ─── Main Application ─────────────────────────────────────────────────────────

def main_app() -> None:
    """Render the main application interface for authenticated users."""
    username = st.session_state.get("username", "Analyst")
    st.markdown(f"""
        <div class="hero-section">
            <div class="hero-badge">✦ AI-Powered Research</div>
            <div class="logo-lockup logo-lockup-center" style="display:flex;justify-content:center;margin-bottom:1.2rem;">
                <div class="siri-orb siri-orb-lg"></div>
                <div class="logo-text-wrap" style="align-items:center;">
                    <div class="hero-title">InsightSphere</div>
                    <p class="hero-subtitle">Ask. Analyze. Invest. &nbsp;|&nbsp; Welcome back, <strong style="color:#f1f5f9">{username}</strong></p>
                </div>
            </div>
        </div>
        <div class="features-grid fade-in">
            <div class="feature-item" style="animation-delay:0.05s">
                <div class="feature-icon">🌐</div>
                <div class="feature-text"><strong>Smart Scraping</strong>Fetches and cleans any public financial news article automatically</div>
            </div>
            <div class="feature-item" style="animation-delay:0.1s">
                <div class="feature-icon">🤖</div>
                <div class="feature-text"><strong>AI Summarization</strong>Extracts key figures, forecasts &amp; events from articles instantly</div>
            </div>
            <div class="feature-item" style="animation-delay:0.15s">
                <div class="feature-icon">🔍</div>
                <div class="feature-text"><strong>Semantic Search</strong>FAISS vector store finds the most relevant context for your query</div>
            </div>
            <div class="feature-item" style="animation-delay:0.2s">
                <div class="feature-icon">💬</div>
                <div class="feature-text"><strong>RAG Q&amp;A</strong>Source-grounded answers — no hallucinations, only facts from articles</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
        <div style="padding:1rem 0 0.5rem 0;">
            <div class="logo-lockup" style="display:flex;align-items:center;gap:0.65rem;">
                <div class="siri-orb siri-orb-sm"></div>
                <div class="logo-text-wrap">
                    <div class="logo-name">InsightSphere</div>
                    <div class="logo-tagline">Financial Research AI</div>
                </div>
            </div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.07);margin:0.8rem 0 1rem 0;">
        <div style="font-size:0.72rem;font-weight:700;color:#64748b;letter-spacing:0.1em;
                    text-transform:uppercase;margin-bottom:0.6rem;">📑 Article URLs</div>
        <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:0.8rem;line-height:1.5;">
            Paste up to 3 financial news URLs<br>
            <span style="color:#4f8ef7;font-size:0.75rem;">Bloomberg · WSJ · Reuters · Moneycontrol</span>
        </div>
    """, unsafe_allow_html=True)

    urls = []
    for i in range(MAX_URLS):
        st.sidebar.markdown(
            f'<div style="font-size:0.7rem;font-weight:600;color:#64748b;letter-spacing:0.08em;'
            f'text-transform:uppercase;margin-bottom:0.3rem;margin-top:{"0" if i==0 else "0.5rem"};">'  
            f'URL {i+1}</div>',
            unsafe_allow_html=True,
        )
        urls.append(st.sidebar.text_input(
            f"url_label_{i}", key=f"url_{i}",
            placeholder="https://...",
            label_visibility="collapsed",
        ))

    st.sidebar.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
    process_clicked = st.sidebar.button("⚡ Process Articles", use_container_width=True)

    st.sidebar.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:1rem 0;'>", unsafe_allow_html=True)
    if st.sidebar.button("🔒 Logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    if process_clicked:
        valid_urls = [url.strip() for url in urls if url and url.startswith("http")]
        if not valid_urls:
            st.markdown("""
                <div style="background:rgba(248,113,113,0.07);border:1px solid rgba(248,113,113,0.2);
                            border-radius:12px;padding:1rem 1.2rem;margin-top:1rem;
                            display:flex;align-items:flex-start;gap:0.75rem;">
                    <div style="font-size:1.3rem;line-height:1;">⚠️</div>
                    <div>
                        <div style="font-weight:600;color:#f87171;font-size:0.9rem;margin-bottom:0.2rem;">No valid URLs</div>
                        <div style="color:#94a3b8;font-size:0.82rem;">Please enter at least one URL starting with http:// or https://</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Show a styled processing status panel
            status_slot = st.empty()
            status_slot.markdown("""
                <div style="background:rgba(79,142,247,0.07);border:1px solid rgba(79,142,247,0.18);
                            border-radius:12px;padding:1.1rem 1.3rem;margin-top:1rem;
                            display:flex;align-items:center;gap:0.85rem;">
                    <div style="font-size:1.4rem;line-height:1;">⏳</div>
                    <div>
                        <div style="font-weight:600;color:#f1f5f9;font-size:0.9rem;margin-bottom:0.15rem;">Processing articles…</div>
                        <div style="color:#64748b;font-size:0.8rem;">Fetching content, summarising, and building vector index</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            try:
                # Clear previous vector store
                if "vectorstore" in st.session_state:
                    del st.session_state["vectorstore"]
                if os.path.exists(VECTOR_STORE_PATH):
                    os.remove(VECTOR_STORE_PATH)

                data = fetch_and_summarize_data_from_urls(valid_urls)
                if not data:
                    status_slot.markdown("""
                        <div style="background:rgba(248,113,113,0.07);border:1px solid rgba(248,113,113,0.2);
                                    border-radius:12px;padding:1rem 1.2rem;margin-top:1rem;
                                    display:flex;align-items:flex-start;gap:0.75rem;">
                            <div style="font-size:1.3rem;line-height:1;">❌</div>
                            <div>
                                <div style="font-weight:600;color:#f87171;font-size:0.9rem;margin-bottom:0.2rem;">No content found</div>
                                <div style="color:#94a3b8;font-size:0.82rem;">The provided URLs returned no readable article content.</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    return

                # Split documents into chunks for embedding
                splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ".", "!", "?", ","],
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=len,
                )
                docs = splitter.split_documents(data)

                # Filter out small or low-quality chunks
                filtered_docs = [
                    doc
                    for doc in docs
                    if len(doc.page_content) > MIN_CHUNK_LENGTH
                    and "subscribe" not in doc.page_content.lower()
                    and "advertisement" not in doc.page_content.lower()
                ]

                if not filtered_docs:
                    status_slot.markdown("""
                        <div style="background:rgba(248,113,113,0.07);border:1px solid rgba(248,113,113,0.2);
                                    border-radius:12px;padding:1rem 1.2rem;margin-top:1rem;
                                    display:flex;align-items:flex-start;gap:0.75rem;">
                            <div style="font-size:1.3rem;line-height:1;">🚫</div>
                            <div>
                                <div style="font-weight:600;color:#f87171;font-size:0.9rem;margin-bottom:0.2rem;">Filtered out</div>
                                <div style="color:#94a3b8;font-size:0.82rem;">No quality content remained after filtering. Try different URLs.</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    return

                embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                vectorstore = FAISS.from_documents(filtered_docs, embeddings)

                with open(VECTOR_STORE_PATH, "wb") as f:
                    pickle.dump(vectorstore, f)

                st.session_state["vectorstore"] = vectorstore
                st.session_state["processed_urls"] = valid_urls

                n = len(valid_urls)
                status_slot.markdown(f"""
                    <div style="background:rgba(52,211,153,0.07);border:1px solid rgba(52,211,153,0.2);
                                border-radius:12px;padding:1.1rem 1.3rem;margin-top:1rem;
                                display:flex;align-items:flex-start;gap:0.85rem;">
                        <div style="font-size:1.4rem;line-height:1;">✅</div>
                        <div>
                            <div style="font-weight:600;color:#34d399;font-size:0.9rem;margin-bottom:0.2rem;">Ready to answer questions</div>
                            <div style="color:#94a3b8;font-size:0.82rem;">{n} article{'s' if n>1 else ''} processed &amp; indexed — scroll down to ask your question.</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                status_slot.markdown(f"""
                    <div style="background:rgba(248,113,113,0.07);border:1px solid rgba(248,113,113,0.2);
                                border-radius:12px;padding:1rem 1.2rem;margin-top:1rem;
                                display:flex;align-items:flex-start;gap:0.75rem;">
                        <div style="font-size:1.3rem;line-height:1;">❌</div>
                        <div>
                            <div style="font-weight:600;color:#f87171;font-size:0.9rem;margin-bottom:0.2rem;">Processing failed</div>
                            <div style="color:#94a3b8;font-size:0.82rem;">{str(e)}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # ── Ask the AI ────────────────────────────────────────────────
    st.markdown("""
        <div style="margin-top:2.2rem;margin-bottom:0.2rem;">
            <div class="section-heading">🧠 Ask the AI</div>
            <div style="font-size:0.82rem;color:#475569;margin-bottom:1rem;margin-top:-0.4rem;">
                Process articles first, then ask any question about their content.
            </div>
        </div>
    """, unsafe_allow_html=True)

    query = st.text_input(
        "Your question",
        key="query_input",
        placeholder="e.g., What were Apple's Q2 earnings results and analyst reactions?",
    )

    if query:
        if os.path.exists(VECTOR_STORE_PATH) or "vectorstore" in st.session_state:
            vectorstore = st.session_state.get("vectorstore")
            if not vectorstore and os.path.exists(VECTOR_STORE_PATH):
                with open(VECTOR_STORE_PATH, "rb") as f:
                    vectorstore = pickle.load(f)

            if vectorstore:
                start = time.time()
                with st.spinner("🔍 Analyzing articles..."):
                    response, sources = answer_question(query, vectorstore)
                end = time.time()

                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:0.8rem;margin:1.2rem 0 0.5rem;">'
                    f'<div class="section-heading" style="margin:0;">💡 Insight</div>'
                    f'<span class="timing-pill">⚡ {end - start:.2f}s</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="answer-box">{response}</div>',
                    unsafe_allow_html=True,
                )

                if sources:
                    st.markdown(
                        '<div style="margin-top:1rem;"><div class="section-heading">📚 Sources</div></div>',
                        unsafe_allow_html=True,
                    )
                    badge_html = "".join(
                        f'<a href="{s}" target="_blank" style="text-decoration:none;">'
                        f'<span class="source-badge">🔗 {s[:60]}{'...' if len(s)>60 else ''}</span></a>'
                        for s in sources
                    )
                    st.markdown(badge_html, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No articles processed yet. Paste URLs in the sidebar and click Process Articles.")


# ─── App Router ───────────────────────────────────────────────────────────────

def app_router() -> None:
    """Route the user to the correct page based on authentication state."""
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "show_register" not in st.session_state:
        st.session_state["show_register"] = False

    if not st.session_state.get("logged_in"):
        if st.session_state.get("show_register"):
            register_page()
        else:
            login_page()
    else:
        main_app()


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    """Application entry point: initialize the database and start the router."""
    create_database()
    app_router()


if __name__ == "__main__":
    main()
