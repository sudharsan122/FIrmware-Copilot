"""
Firmware Copilot — Professional WhatsApp-style chat UI (Refactored for Streamlit)
RAG system for driver code generation from PDF documents.
"""

import os
import sys
import io
import re
import numpy as np
from datetime import datetime
import chromadb
from chromadb.config import Settings

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openai.generative.engine.capgemini.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")

if not API_KEY:
    # Ignore a missing key during startup so the app can load and show a clear prompt
    # instead of crashing during import time.
    load_dotenv(override=True)
    API_KEY = os.getenv("OPENAI_API_KEY", "")
else:
    load_dotenv(override=True)
    API_KEY = os.getenv("OPENAI_API_KEY", API_KEY)

client = OpenAI(base_url=BASE_URL, api_key=API_KEY) if API_KEY else None

# =============================================
# VECTOR DATABASE SETUP — Chroma Integration
# =============================================
CHROMA_DATA_DIR = ".chroma_data/"
chroma_settings = Settings(
    is_persistent=True,
    persist_directory=CHROMA_DATA_DIR,
    anonymized_telemetry=False
)

try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_DIR)
except (AttributeError, TypeError):
    chroma_client = chromadb.Client(chroma_settings)

# =============================================
# CHROMA VECTOR DB CLASS
# =============================================
class ChromaVectorDB:
    """Wrapper for Chroma vector database operations."""
    
    def __init__(self, client):
        self.client = client
    
    @staticmethod
    def sanitize_collection_name(name: str) -> str:
        """Sanitize collection name to meet Chroma requirements."""
        # Remove .pdf extension and replace invalid characters
        name = name.replace(".pdf", "").replace(".PDF", "")
        # Replace spaces and special chars with underscores
        name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        # Ensure it starts with a letter or underscore
        if name and not name[0].isalpha() and name[0] != '_':
            name = '_' + name
        # Limit to 63 characters
        return name[:63] if name else "collection"
    
    def create_or_get_collection(self, collection_name: str):
        """Create or retrieve a collection by name."""
        safe_name = self.sanitize_collection_name(collection_name)
        return self.client.get_or_create_collection(
            name=safe_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, collection, chunks: list, chunk_ids: list = None):
        """Add text chunks to collection. Chroma auto-generates embeddings."""
        if not chunks:
            return
        
        if chunk_ids is None:
            chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        collection.add(
            ids=chunk_ids,
            documents=chunks,
            metadatas=[{"source": "pdf", "chunk_index": i} for i in range(len(chunks))]
        )
    
    def search(self, collection, query: str, top_k: int = 3) -> list:
        """Search for similar chunks using semantic similarity."""
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        if results and results['documents']:
            return results['documents'][0]  # Return list of matching chunks
        return []
    
    def list_collections(self) -> list:
        """List all available collections."""
        return self.client.list_collections()
    
    def delete_collection(self, collection_name: str):
        """Delete a collection by name."""
        safe_name = self.sanitize_collection_name(collection_name)
        try:
            self.client.delete_collection(name=safe_name)
            return True
        except:
            return False
    
    def get_collection_info(self, collection_name: str) -> dict:
        """Get metadata about a collection."""
        safe_name = self.sanitize_collection_name(collection_name)
        try:
            collection = self.client.get_collection(name=safe_name)
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except:
            return {}

# Initialize ChromaVectorDB
vector_db = ChromaVectorDB(chroma_client)

name = "Firmware Copilot"
#PDF = "document.pdf"

# ---------- Brand tokens ----------
BRAND_GREEN = "#00FF00"
BRAND_PURPLE = "#00CC66"
BRAND_PINK = "#000000"
BRAND_GRADIENT = "linear-gradient(135deg, #000000 0%, #001b0d 25%, #00FF00 100%)"


# ---------- Page config ----------
st.set_page_config(
    page_title="Firmware Copilot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------- Global styling ----------
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            /* ---------- Base ---------- */
            html, body, [data-testid="stAppViewContainer"], .main, .block-container {{
                background: #000000 !important;
                color: {BRAND_GREEN} !important;
            }}
            [data-testid="stHeader"] {{
                background: transparent !important;
            }}
            .block-container {{
                padding-top: 1.2rem;
                padding-bottom: 7rem;
                max-width: 820px;
            }}
            * {{
                font-family: 'Consolas', 'Cascadia Code', 'Courier New', monospace;
                scrollbar-color: {BRAND_GREEN} #000000;
            }}
            ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
            ::-webkit-scrollbar-track {{ background: #000000; }}
            ::-webkit-scrollbar-thumb {{ background: {BRAND_GREEN}; border-radius: 999px; }}

            /* ---------- Sidebar ---------- */
            [data-testid="stSidebar"] {{
                background: #000000 !important;
                border-right: 1px solid {BRAND_GREEN};
                box-shadow: inset -1px 0 0 rgba(0,255,0,0.25);
            }}
            [data-testid="stSidebar"] .sidebar-brand {{
                display: flex; align-items: center; gap: 12px;
                padding: 8px 4px 18px 4px;
                border-bottom: 1px solid rgba(0,255,0,0.4);
                margin-bottom: 16px;
            }}
            [data-testid="stSidebar"] .sidebar-brand .logo {{
                width: 38px; height: 38px; border-radius: 12px;
                background: {BRAND_GRADIENT};
                box-shadow: 0 0 0 1px rgba(0,255,0,0.8), 0 0 16px rgba(0,255,0,0.35);
            }}
            [data-testid="stSidebar"] .sidebar-brand .brand-name {{
                font-weight: 700; font-size: 17px; color: {BRAND_GREEN};
                letter-spacing: -0.01em;
            }}
            [data-testid="stSidebar"] .sidebar-brand .brand-tag {{
                font-size: 11.5px; color: rgba(0,255,0,0.8); margin-top: 2px;
            }}
            [data-testid="stSidebar"] .side-section-title {{
                text-transform: uppercase; letter-spacing: 0.08em;
                font-size: 11px; color: {BRAND_GREEN}; font-weight: 700;
                margin: 14px 0 8px 0;
                text-shadow: 0 0 8px rgba(0,255,0,0.4);
            }}
            [data-testid="stSidebar"] .side-card {{
                background: #00140a;
                border: 1px solid {BRAND_GREEN};
                border-radius: 14px;
                padding: 12px 14px;
                font-size: 13px; color: {BRAND_GREEN};
                line-height: 1.5;
                box-shadow: inset 0 0 0 1px rgba(0,255,0,0.15), 0 0 12px rgba(0,255,0,0.10);
            }}
            .about-typing-line {{
                display: block;
                width: max-content;
                max-width: 100%;
                overflow: hidden;
                white-space: nowrap;
                clip-path: inset(0 100% 0 0);
                animation-duration: 20s;
                animation-iteration-count: infinite;
                animation-timing-function: linear;
            }}
            .about-typing-line:nth-child(1) {{
                animation-name: about-line-one;
            }}
            .about-typing-line:nth-child(2) {{
                animation-name: about-line-two;
            }}
            .about-typing-line:nth-child(3) {{
                animation-name: about-line-three;
            }}
            .about-typing-line:nth-child(4) {{
                animation-name: about-line-four;
            }}
            .about-typing-line:nth-child(5) {{
                animation-name: about-line-five;
            }}
            @keyframes about-line-one {{
                0% {{ clip-path: inset(0 100% 0 0); animation-timing-function: steps(31, end); }}
                10%, 91% {{ clip-path: inset(0 0 0 0); }}
                92% {{ clip-path: inset(0 0 0 0); animation-timing-function: steps(31, end); }}
                100% {{ clip-path: inset(0 100% 0 0); }}
            }}
            @keyframes about-line-two {{
                0%, 10% {{ clip-path: inset(0 100% 0 0); }}
                10% {{ animation-timing-function: steps(29, end); }}
                20%, 83% {{ clip-path: inset(0 0 0 0); }}
                84% {{ clip-path: inset(0 0 0 0); animation-timing-function: steps(29, end); }}
                92%, 100% {{ clip-path: inset(0 100% 0 0); }}
            }}
            @keyframes about-line-three {{
                0%, 20% {{ clip-path: inset(0 100% 0 0); }}
                20% {{ animation-timing-function: steps(26, end); }}
                30%, 75% {{ clip-path: inset(0 0 0 0); }}
                76% {{ clip-path: inset(0 0 0 0); animation-timing-function: steps(26, end); }}
                84%, 100% {{ clip-path: inset(0 100% 0 0); }}
            }}
            @keyframes about-line-four {{
                0%, 30% {{ clip-path: inset(0 100% 0 0); }}
                30% {{ animation-timing-function: steps(26, end); }}
                40%, 67% {{ clip-path: inset(0 0 0 0); }}
                68% {{ clip-path: inset(0 0 0 0); animation-timing-function: steps(26, end); }}
                76%, 100% {{ clip-path: inset(0 100% 0 0); }}
            }}
            @keyframes about-line-five {{
                0%, 40% {{ clip-path: inset(0 100% 0 0); }}
                40% {{ animation-timing-function: steps(27, end); }}
                50%, 59% {{ clip-path: inset(0 0 0 0); }}
                60% {{ clip-path: inset(0 0 0 0); animation-timing-function: steps(27, end); }}
                68%, 100% {{ clip-path: inset(0 100% 0 0); }}
            }}
            [data-testid="stSidebar"] .pill {{
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                background: rgba(0,255,0,0.08);
                color: {BRAND_GREEN};
                border: 1px solid rgba(0,255,0,0.45);
                font-size: 11.5px; font-weight: 700;
                margin-right: 6px; margin-top: 6px;
            }}
            [data-testid="stSidebar"] .pill.purple {{
                background: rgba(0,255,0,0.06); color: #7DFFB5; border-color: #7DFFB5;
            }}
            [data-testid="stSidebar"] .pill.pink {{
                background: rgba(0,255,0,0.06); color: #9AFFA8; border-color: #9AFFA8;
            }}

            /* Sidebar button */
            [data-testid="stSidebar"] .stButton > button {{
                width: 100%;
                border-radius: 12px;
                border: 1px solid {BRAND_GREEN};
                background: #000000;
                color: {BRAND_GREEN};
                font-weight: 700; font-size: 13px;
                padding: 10px 12px;
                transition: all 0.15s ease;
                box-shadow: 0 0 0 1px rgba(0,255,0,0.15);
            }}
            [data-testid="stSidebar"] .stButton > button:hover {{
                border-color: {BRAND_GREEN};
                background: rgba(0,255,0,0.12);
                color: #AAFFAA;
                box-shadow: 0 0 12px rgba(0,255,0,0.28);
            }}
            [data-testid="stSidebar"] .stButton > button:focus {{
                box-shadow: 0 0 0 2px rgba(0,255,0,0.35), 0 0 10px rgba(0,255,0,0.25);
                outline: none;
            }}
            [data-testid="stSidebar"] [data-testid="column"] .stButton > button {{
                min-height: 96px;
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow-wrap: anywhere;
            }}
            [data-testid="stSidebar"] [data-testid="column"]:last-child .stButton > button p {{
                color: {BRAND_GREEN} !important;
                filter: grayscale(1) sepia(1) saturate(8) hue-rotate(65deg) brightness(1.15);
                text-shadow: 0 0 8px rgba(0,255,0,0.7);
            }}

            /* ---------- PDF uploader ---------- */
            [data-testid="stSidebar"] label {{
                color: {BRAND_GREEN} !important;
            }}
            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
                background: #000000 !important;
                border: 1px dashed {BRAND_GREEN} !important;
                border-radius: 12px !important;
            }}
            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div {{
                background: #000000 !important;
                color: {BRAND_GREEN} !important;
            }}
            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{
                background: #00140a !important;
                color: {BRAND_GREEN} !important;
                border: 1px solid {BRAND_GREEN} !important;
                box-shadow: 0 0 8px rgba(0,255,0,0.16) !important;
            }}
            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {{
                background: rgba(0,255,0,0.12) !important;
                box-shadow: 0 0 12px rgba(0,255,0,0.3) !important;
            }}
            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] svg {{
                color: {BRAND_GREEN} !important;
                fill: {BRAND_GREEN};
            }}

            /* ---------- Cyberpunk title ---------- */
            .cyber-title {{
                text-align: center;
                margin: 0 auto 12px auto;
                padding: 10px 0 8px 0;
                animation: title-fade-in 900ms ease-out both;
            }}
            .cyber-title h1 {{
                color: #00FF41;
                font-size: clamp(38px, 7vw, 60px);
                font-weight: 800;
                line-height: 1.08;
                letter-spacing: 0.015em;
                margin: 0;
                text-shadow: 0 0 8px rgba(0,255,65,0.55), 0 0 24px rgba(0,255,65,0.24);
                animation: title-flicker 7s infinite;
            }}
            .author-signature {{
                display: inline-block;
                position: relative;
                margin-top: 7px;
                color: rgba(0,255,65,0.78);
                font-size: clamp(15px, 2.4vw, 19px);
                letter-spacing: 0.12em;
                text-shadow: 1px 0 rgba(0,255,65,0.48), -1px 0 rgba(0,180,55,0.32);
                animation: author-glitch 3.6s infinite steps(1, end);
            }}
            .author-signature::before,
            .author-signature::after {{
                content: attr(data-text);
                position: absolute;
                inset: 0;
                pointer-events: none;
            }}
            .author-signature::before {{
                color: rgba(0,255,65,0.55);
                animation: author-glitch-before 2.8s infinite steps(1, end);
            }}
            .author-signature::after {{
                color: rgba(0,180,55,0.62);
                animation: author-glitch-after 3.2s infinite steps(1, end);
            }}
            @keyframes title-fade-in {{
                from {{ opacity: 0; transform: translateY(-8px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            @keyframes title-flicker {{
                0%, 96%, 98%, 100% {{ opacity: 1; }}
                97% {{ opacity: 0.86; }}
                99% {{ opacity: 0.94; }}
            }}
            @keyframes author-glitch {{
                0%, 94%, 96%, 100% {{ transform: translateX(0); }}
                95% {{ transform: translateX(1px); }}
                97% {{ transform: translateX(-1px); }}
            }}
            @keyframes author-glitch-before {{
                0%, 90%, 94%, 100% {{ clip-path: inset(0 0 0 0); transform: translateX(0); }}
                91% {{ clip-path: inset(15% 0 55% 0); transform: translateX(-2px); }}
                93% {{ clip-path: inset(62% 0 12% 0); transform: translateX(2px); }}
            }}
            @keyframes author-glitch-after {{
                0%, 88%, 92%, 100% {{ clip-path: inset(0 0 0 0); transform: translateX(0); }}
                89% {{ clip-path: inset(48% 0 28% 0); transform: translateX(2px); }}
                91% {{ clip-path: inset(8% 0 72% 0); transform: translateX(-2px); }}
            }}

            /* ---------- Chat area ---------- */
            .chat-area {{
                background: #000000;
                padding: 8px 4px 24px 4px;
            }}
            .day-divider {{
                text-align: center;
                margin: 18px 0 14px 0;
            }}
            .day-divider span {{
                background: #00140a;
                color: {BRAND_GREEN};
                border: 1px solid rgba(0,255,0,0.35);
                font-size: 11.5px; font-weight: 700;
                padding: 5px 12px;
                border-radius: 999px;
                box-shadow: 0 0 10px rgba(0,255,0,0.12);
            }}

            /* Bubble rows */
            .row {{
                display: flex; width: 100%; margin: 6px 0;
            }}
            .row.user {{ justify-content: flex-end; }}
            .row.bot  {{ justify-content: flex-start; }}

            .bubble {{
                max-width: 78%;
                padding: 10px 14px 8px 14px;
                font-size: 14.5px; line-height: 1.5;
                position: relative;
                word-wrap: break-word;
                box-shadow: 0 0 12px rgba(0,255,0,0.08);
            }}
            .bubble .time {{
                display: block;
                font-size: 10.5px;
                margin-top: 4px;
                opacity: 0.8;
                text-align: right;
            }}

            /* User bubble */
            .bubble.user {{
                background: rgba(0,255,0,0.12);
                color: #B6FFB6;
                border: 1px solid {BRAND_GREEN};
                border-radius: 16px 16px 4px 16px;
                box-shadow: 0 0 16px rgba(0,255,0,0.10);
            }}
            .bubble.user::after {{
                content: "";
                position: absolute;
                right: -6px; bottom: 0;
                width: 12px; height: 14px;
                background: rgba(0,255,0,0.12);
                border-right: 1px solid {BRAND_GREEN};
                border-bottom: 1px solid {BRAND_GREEN};
                clip-path: polygon(0 0, 0% 100%, 100% 100%);
                border-bottom-right-radius: 2px;
            }}
            .bubble.user .time {{ color: rgba(0,255,0,0.75); }}

            /* Bot bubble */
            .bubble.bot {{
                background: #00140a;
                color: {BRAND_GREEN};
                border-radius: 16px 16px 16px 4px;
                border: 1px solid rgba(0,255,0,0.45);
                border-left: 3px solid {BRAND_GREEN};
                box-shadow: 0 0 12px rgba(0,255,0,0.05);
            }}
            .bubble.bot::after {{
                content: "";
                position: absolute;
                left: -6px; bottom: 0;
                width: 12px; height: 14px;
                background: #00140a;
                border-left: 1px solid rgba(0,255,0,0.45);
                border-bottom: 1px solid rgba(0,255,0,0.45);
                clip-path: polygon(100% 0, 0% 100%, 100% 100%);
                border-bottom-left-radius: 2px;
            }}
            .bubble.bot .time {{ color: rgba(0,255,0,0.7); }}

            /* Typing indicator */
            .typing {{
                display: inline-flex; align-items: center; gap: 4px;
                padding: 6px 2px;
            }}
            .typing span {{
                width: 7px; height: 7px; border-radius: 50%;
                background: {BRAND_GREEN};
                opacity: 0.5;
                animation: blink 1.2s infinite ease-in-out;
                box-shadow: 0 0 10px rgba(0,255,0,0.35);
            }}
            .typing span:nth-child(2) {{ background: #7DFFB5; animation-delay: 0.15s; }}
            .typing span:nth-child(3) {{ background: {BRAND_GREEN}; animation-delay: 0.3s; }}
            @keyframes blink {{
                0%, 80%, 100% {{ opacity: 0.25; transform: translateY(0); }}
                40%           {{ opacity: 1;    transform: translateY(-2px); }}
            }}

            /* Empty state */
            .empty-state {{
                text-align: center;
                padding: 23px 14px 24px 14px;
                color: {BRAND_GREEN};
            }}
            .empty-state .badge {{
                width: 64px; height: 64px; border-radius: 20px;
                background: {BRAND_GRADIENT};
                display: inline-flex; align-items: center; justify-content: center;
                color: #000000; font-size: 28px;
                box-shadow: 0 0 18px rgba(0,255,0,0.3);
                margin-bottom: 14px;
            }}
            .empty-state h3 {{
                color: #00FF41; margin: 0 0 10px 0;
                font-size: clamp(24px, 4vw, 32px); font-weight: 800;
                letter-spacing: 0.035em;
                text-shadow: 0 0 8px rgba(0,255,65,0.48), 0 0 18px rgba(0,255,65,0.2);
                animation: welcome-fade-in 1.1s ease-out both;
            }}
            .empty-state p {{
                font-size: 13.5px; line-height: 1.7;
                margin: 0 0 18px 0; color: rgba(0,255,0,0.8);
            }}
            .terminal-copy {{
                display: block;
                width: max-content;
                max-width: 100%;
                margin-left: auto;
                margin-right: auto;
                overflow: hidden;
                white-space: nowrap;
                clip-path: inset(0 100% 0 0);
            }}
            .terminal-line-one {{
                animation: terminal-type-first 4s steps(69, end) 500ms forwards;
            }}
            .terminal-line-two {{
                animation: terminal-type-second 4s steps(74, end) 4500ms forwards;
            }}
            .terminal-cursor {{
                display: inline-block;
                margin-left: 3px;
                color: {BRAND_GREEN};
                animation: cursor-blink 850ms steps(1, end) infinite;
            }}
            @keyframes welcome-fade-in {{
                from {{ opacity: 0; transform: translateY(8px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            @keyframes terminal-type-first {{
                to {{ clip-path: inset(0 0 0 0); }}
            }}
            @keyframes terminal-type-second {{
                to {{ clip-path: inset(0 0 0 0); }}
            }}
            @keyframes cursor-blink {{
                0%, 50% {{ opacity: 1; }}
                51%, 100% {{ opacity: 0; }}
            }}
            .suggestions {{
                display: flex; flex-wrap: wrap; gap: 8px;
                justify-content: center;
                margin-top: 6px;
            }}
            .suggestion {{
                font-size: 12.5px;
                color: {BRAND_GREEN};
                background: #000000;
                border: 1px solid rgba(0,255,0,0.45);
                padding: 8px 14px;
                border-radius: 999px;
                box-shadow: inset 0 0 0 1px rgba(0,255,0,0.15);
            }}

            /* ---------- Composer (chat input) ---------- */
            [data-testid="stBottom"],
            [data-testid="stBottom"] > div,
            [data-testid="stBottomBlockContainer"],
            [data-testid="stChatInputContainer"],
            [data-testid="stChatInput"],
            [data-testid="stChatInput"] > div,
            [data-testid="stChatInput"] > div > div {{
                background: #000000 !important;
            }}
            [data-testid="stBottom"] {{
                border-top: 1px solid rgba(0,255,0,0.45) !important;
                box-shadow: 0 -8px 24px rgba(0,255,0,0.08) !important;
            }}
            [data-testid="stBottomBlockContainer"] {{
                padding-top: 14px !important;
                padding-bottom: 14px !important;
                max-width: 820px;
            }}
            [data-testid="stChatInput"] textarea {{
                background: #000000 !important;
                border: 2px solid {BRAND_GREEN} !important;
                border-radius: 16px !important;
                color: {BRAND_GREEN} !important;
                caret-color: {BRAND_GREEN} !important;
                padding: 12px 14px !important;
                font-size: 14.5px !important;
                flex: 1 1 0% !important;
                width: 0 !important;
                min-width: 0 !important;
                margin-right: 0 !important;
                box-shadow: inset 0 0 12px rgba(0,255,0,0.06), 0 0 0 1px rgba(0,255,0,0.16);
            }}
            [data-testid="stChatInput"] textarea::placeholder {{
                color: rgba(0,255,0,0.72) !important;
                opacity: 1 !important;
            }}
            [data-testid="stChatInput"] textarea:focus {{
                border-color: #A5FFB5 !important;
                box-shadow: 0 0 0 3px rgba(0,255,0,0.15), inset 0 0 12px rgba(0,255,0,0.08) !important;
            }}
            [data-testid="stChatInput"] button {{
                background: #00140a !important;
                border: 1px solid {BRAND_GREEN} !important;
                border-radius: 12px !important;
                color: {BRAND_GREEN} !important;
                box-shadow: 0 0 14px rgba(0,255,0,0.22) !important;
                font-weight: 700;
            }}
            [data-testid="stChatInput"] button:hover {{
                background: rgba(0,255,0,0.14) !important;
                box-shadow: 0 0 18px rgba(0,255,0,0.42) !important;
            }}
            [data-testid="stChatInput"] button:focus {{
                box-shadow: 0 0 0 2px rgba(0,255,0,0.25), 0 0 18px rgba(0,255,0,0.42) !important;
            }}
            [data-testid="stChatInput"] button svg {{
                color: {BRAND_GREEN} !important;
                fill: {BRAND_GREEN} !important;
            }}
            [data-testid="stChatInput"] > div {{
                position: relative;
                box-sizing: border-box;
                padding-right: 54px !important;
            }}
            [data-testid="stChatInput"]:has(textarea:placeholder-shown:not(:focus)) > div::after {{
                content: "|";
                position: absolute;
                left: calc(14px + 17ch);
                top: 50%;
                transform: translateY(-50%);
                color: {BRAND_GREEN};
                font-weight: 700;
                pointer-events: none;
                animation: cursor-blink 850ms steps(1, end) infinite;
            }}

            /* Hide default Streamlit chrome */
            #MainMenu, footer {{ visibility: hidden; }}

            /* Spinner color */
            .stSpinner > div > div {{
                border-top-color: {BRAND_GREEN} !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------- UI helpers ----------
def render_header() -> None:
    st.markdown(
        """
        <div class="cyber-title">
            <h1>Firmware Copilot</h1>
            <span class="author-signature" data-text="by Sudharsan S">by Sudharsan S</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> any:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="logo"></div>
                <div>
                    <div class="brand-name">Firmware Copilot</div>
                    <div class="brand-tag">AI Companion Platform</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="side-section-title">About</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="side-card">
                <span class="about-typing-line">Ask this copilot anything about</span>
                <span class="about-typing-line">your Manual to get answers or</span>
                <span class="about-typing-line">generate code. A companion</span>
                <span class="about-typing-line">developed by Sudharsan S -</span>
                <span class="about-typing-line">Embedded Software Engineer.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="side-section-title">PDF Upload</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

        # ===== COLLECTIONS MANAGEMENT =====
        st.markdown('<div class="side-section-title"> Collections</div>', unsafe_allow_html=True)
        
        collections = vector_db.list_collections()
        if collections:
            st.markdown(
                f"""
                <div class="side-card">
                    <strong>{len(collections)} collection(s) loaded</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            for col in collections:
                col_info = vector_db.get_collection_info(col.name)
                col_name = col.name
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if st.button(f"✓ {col_name}", use_container_width=True):
                        st.session_state.current_collection = col_name
                        st.session_state.current_collection_obj = col
                        st.rerun()
                
                with col2:
                    if st.button("⌫", key=f"del_{col_name}"):
                        vector_db.delete_collection(col_name)
                        st.session_state.current_collection = None
                        st.session_state.current_collection_obj = None
                        st.rerun()
        else:
            st.markdown(
                """
                <div class="side-card">
                    No collections yet. Upload a PDF to create one.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="side-section-title">Session</div>', unsafe_allow_html=True)
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown(
            """
            <div style="position:absolute; bottom:18px; left:18px; right:18px;
                        font-size:11px; color:#9B9BAC; text-align:center;">
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        return uploaded_file


def now_hm() -> str:
    return datetime.now().strftime("%I:%M %p").lstrip("0")


def render_message(role: str, text: str, ts: str) -> None:
    """role ∈ {'user', 'bot'}"""
    safe = (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
    )
    st.markdown(
        f"""
        <div class="row {role}">
            <div class="bubble {role}">
                {safe}
                <span class="time">{ts}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <h3>FIRMWARE COPILOT</h3>
            <p><span class="terminal-copy terminal-line-one">Ask me anything about your manual. I can answer your queries and also</span><span class="terminal-copy terminal-line-two">generate .h and .c driver files directly from your uploaded documentation.</span><span class="terminal-cursor">█</span></p>
            <div class="suggestions">
                <div class="suggestion">Generate the code</div>
                <div class="suggestion">Generate</div>
                <div class="suggestion">Generate from manual</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ========================
# STEP 2 — RAG : Chunking
# ========================
def chunk_text(text, chunk_size=1000, overlap=50):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# =========================
# STEP 3 — RAG : Embedding
# =========================
def get_embeddings(texts):
    """Convert a list of strings into embedding vectors."""
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your environment or .env file before chatting.")
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([item.embedding for item in response.data], dtype=np.float32)


# ========================
# STEP 4 — RAG : Retrieval
# ========================
def cosine_formula(vec_1, vec_2):
    dot = 0
    for a, b in zip(vec_1, vec_2):
        dot += (a * b)

    vec_1_len = 0
    for a in vec_1:
        vec_1_len += (a * a)
    vec_1_len = vec_1_len ** 0.5

    vec_2_len = 0
    for b in vec_2:
        vec_2_len += (b * b)
    vec_2_len = vec_2_len ** 0.5

    denominator = vec_1_len * vec_2_len
    if denominator == 0:
        return 0.0

    similarity = dot / denominator
    return similarity


def retrieve(query, chunks, chunk_embeddings, top_k=3):
    """Find the most relevant chunks for a given question."""
    if chunk_embeddings is None or len(chunks) == 0:
        return []

    query_vec = get_embeddings([query])[0]

    all_score = []
    for chunk_vec in chunk_embeddings:
        score = cosine_formula(query_vec, chunk_vec)
        all_score.append(score)

    indx = list(range(len(all_score)))
    indx.sort(key=lambda i: all_score[i], reverse=True)

    top_k_res = []
    for i in indx[:top_k]:
        top_k_res.append(chunks[i])

    return top_k_res


# ===========================================
# STEP 5 — PROMPT ENGINEERING : System Prompt
# ===========================================
system_prompt = f"""You are acting as {name}.
        You are an embedded systems expert, driver generator, and GPIO/Hardware consultant.
 
        =======================
        CORE RESPONSIBILITIES
        =======================
        1. Answer educational questions about GPIO, drivers, and embedded systems using PDF context
        2. Generate driver code when user explicitly requests it (keywords: "generate", "create", "code", "driver")
        3. Explain hardware concepts, register details, and configuration methods
        4. Use ONLY information from the provided PDF context
        5. DO NOT invent registers, macros, or hardware behavior not in the PDF
 
        =======================
        ANSWER TYPES YOU SUPPORT
        =======================
        
        TYPE 1 — EDUCATIONAL QUESTIONS
        Keywords: "what is", "explain", "how to", "configure", "setup", "tell me about"
        Action: Answer using PDF context with clear explanations
        Example: "what is GPIO and how to configure it?"
        → Provide explanation from PDF context
        
        TYPE 2 — DRIVER CODE GENERATION
        Keywords: "generate", "create", "code", "write driver", "make a driver"
        Action: Generate .h and .c driver files
        Example: "generate GPIO driver for PIC16F62xA"
        → Output:
           // filename.h
           <code>
           // filename.c
           <code>
        
        TYPE 3 — REFERENCE QUESTIONS
        Keywords: "register", "pin", "datasheet", "chapter", "section"
        Action: Provide specific register/pin information from PDF
        Example: "what is GPIO register for PIC16F62xA?"
        → Explain from PDF context
 
        =======================
        VALIDATION LOGIC
        =======================
        - If question is about GPIO/drivers: Answer using PDF context
        - If PDF context is missing: Respond "INSUFFICIENT DATA FROM PDF"
        - If user requests driver generation: Provide code in required format
        - If unclear request: Ask for clarification instead of rejecting
 
        =======================
        FILE NAMING RULE (FOR CODE GENERATION)
        =======================
        You MUST dynamically generate file names using:
 
        <device>_<driverType>_ch<chapter>
 
        Examples:
        - eeprom_i2c_ch5.h
        - adc_spi_ch3.c
        - temp_uart_ch2.h
 
        Rules:
        - device → from user input (fallback: "device")
        - driverType → lowercase (i2c/spi/uart/gpio)
        - chapter → number extracted from context/user input
 
        =======================
        OUTPUT FORMAT
        =======================
        ONLY IF ALL VALIDATIONS PASS:
 
        // <generated_file_name>.h
        <code>
 
        // <generated_file_name>.c
        <code>
 
        Do NOT output anything else.

        """


# =======================================================
# STEP 6 — PROMPT ENGINEERING : User Message with Context
# =======================================================
def build_user_message(question, relevant_chunks):
    """
    Wrap the user's question with retrieved context.
    This is the 'Augmented' part of RAG.
    """
    context = " ".join(relevant_chunks)
    return f"""[Relevant info from document]
    {context}
    
    [Question]
    {question}
    
    Answer based only on the provided info above."""


# =============================================
# HELPER: Load and cache PDF processing
# =============================================
@st.cache_resource(show_spinner="Loading document...")
def load_pdf(uploaded_file):
    """Load PDF, extract text into chunks, and create Chroma collection."""
    if uploaded_file is None:
        return "", [], None
    
    pdf_bytes = io.BytesIO(uploaded_file.read())
    reader = PdfReader(pdf_bytes)
    
    profile = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            profile += text
    
    chunks = chunk_text(profile)
    
    # Create Chroma collection for this PDF
    collection_name = uploaded_file.name
    collection = vector_db.create_or_get_collection(collection_name)
    
    # Add chunks to Chroma (auto-generates embeddings)
    vector_db.add_documents(collection, chunks)
    
    return profile, chunks, collection


@st.cache_resource(show_spinner="Creating embeddings...")
def prepare_embeddings(chunks):
    """Generate embeddings for all chunks."""
    if not chunks:
        return None
    return get_embeddings(chunks)


def stream_response(messages: list) -> str:
    """Stream response from OpenAI API and return complete answer."""
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your environment or .env file before chatting.")

    complete_answer = ""
    response_placeholder = st.empty()
    
    try:
        stream = client.chat.completions.create(
            model="openai.gpt-5-nano",
            messages=messages,
            temperature=0.3,
            max_tokens=300,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                complete_answer += chunk.choices[0].delta.content
                
                # Update UI in real-time with streaming text
                response_placeholder.markdown(
                    f"""
                    <div class="row bot">
                        <div class="bubble bot">
                            {complete_answer.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(chr(10), "<br>")}
                            <span class="time">{now_hm()}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        
        return complete_answer
    
    except Exception as e:
        error_msg = f"Sorry, I hit an error while answering: {e}"
        response_placeholder.markdown(
            f"""
            <div class="row bot">
                <div class="bubble bot">
                    {error_msg}
                    <span class="time">{now_hm()}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return error_msg


def retrieve_from_vector_db(query: str, collection, top_k: int = 3) -> list:
    """Retrieve relevant chunks from Chroma collection using semantic search."""
    if collection is None:
        return []
    return vector_db.search(collection, query, top_k)


# =============================================
# MAIN STREAMLIT APPLICATION
# =============================================
def main() -> None:
    inject_css()
    uploaded_file = render_sidebar()
    render_header()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "profile" not in st.session_state:
        st.session_state.profile = ""
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "chunk_embeddings" not in st.session_state:
        st.session_state.chunk_embeddings = None
    if "current_collection" not in st.session_state:
        st.session_state.current_collection = None
    if "current_collection_obj" not in st.session_state:
        st.session_state.current_collection_obj = None

    # Load PDF if uploaded
    if uploaded_file is not None:
        st.session_state.profile, st.session_state.chunks, collection = load_pdf(uploaded_file)
        st.session_state.current_collection = uploaded_file.name
        st.session_state.current_collection_obj = collection
        st.sidebar.success(f"✅ Loaded {len(st.session_state.chunks)} chunks from PDF")

    # Chat area
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="day-divider"><span>{datetime.now().strftime("%A, %d %b %Y")}</span></div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.messages:
            render_empty_state()
        else:
            for m in st.session_state.messages:
                render_message(m["role"], m["text"], m["ts"])

        st.markdown("</div>", unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Type a message…")
    if user_input:
        if st.session_state.current_collection_obj is None:
            st.error("⚠️ Please upload a PDF first!")
        else:
            # Add user message
            st.session_state.messages.append(
                {"role": "user", "text": user_input, "ts": now_hm()}
            )
            render_message("user", user_input, now_hm())

            # Show typing indicator
            typing_slot = st.empty()
            typing_slot.markdown(
                """
                <div class="row bot">
                    <div class="bubble bot">
                        <div class="typing"><span></span><span></span><span></span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            try:
                # RAG: Retrieve relevant chunks from Chroma
                relevant = retrieve_from_vector_db(
                    user_input,
                    st.session_state.current_collection_obj,
                    top_k=3
                )
                
                # Build augmented message with context
                augmented_message = build_user_message(user_input, relevant)

                # Prepare messages for API
                history = [{"role": "system", "content": system_prompt}]
                
                # Add conversation history (excluding current user message)
                for msg in st.session_state.messages[:-1]:
                    history.append({"role": msg["role"], "content": msg["text"]})
                
                # Add current augmented message
                history.append({"role": "user", "content": augmented_message})

                # Stream response from OpenAI (Copilot-like real-time generation)
                typing_slot.empty()
                answer = stream_response(history)

            except Exception as exc:
                typing_slot.empty()
                answer = f"Sorry, I hit an error while answering: {exc}"
                st.markdown(
                    f"""
                    <div class="row bot">
                        <div class="bubble bot">
                            {str(answer).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(chr(10), "<br>")}
                            <span class="time">{now_hm()}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Add bot response to session
            st.session_state.messages.append(
                {"role": "assistant", "text": str(answer), "ts": now_hm()}
            )

            # Handle file generation (only if code is present in response)
            output_dir = os.path.join(os.getcwd(), "generated_drivers")
            os.makedirs(output_dir, exist_ok=True)

            # Look for code files in format: // filename.h or // filename.c
            pattern = r"//\s*(\S+\.(h|c))\n([\s\S]*?)(?=(//\s*\S+\.(h|c))|\Z)"
            matches = re.findall(pattern, answer)

            if matches:
                for match in matches:
                    filename = match[0]
                    code = match[2]

                    driver_folder = filename.split(".")[0]
                    driver_dir = os.path.join(output_dir, driver_folder)
                    os.makedirs(driver_dir, exist_ok=True)

                    file_path = os.path.join(driver_dir, filename)

                    with open(file_path, "w") as f:
                        f.write(code.strip())

                    st.success(f"✅ File created: {file_path}")
            else:
                # No code files to generate (educational question)
                pass

            st.rerun()


if __name__ == "__main__":
    main()
