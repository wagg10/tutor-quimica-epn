from pathlib import Path
import re
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# =========================
# CONFIGURACIÓN DEL SISTEMA
# =========================

BASE_DIR = Path.home() / "Quimica" / "rag_quimica_general"
DB_PATH = BASE_DIR / "bd_semantica_quimica"
COLLECTION_NAME = "coleccion_libro_quimica_general"

SKILL_PATHS = [
    Path.home() / "Quimica" / "skills" / "tutor-quimica-epn-runtime.md",
    Path.home() / "Quimica" / "skills" / "tutor-quimica-epn-SKILL.md",
]

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

LLM_BASE_URL = "http://127.0.0.1:1234/v1"
LLM_API_KEY = "lm-studio"
LLM_MODEL = "phi4-quimica-v2"

N_FRAGMENTOS = 3


# =========================
# ESTILO VISUAL
# =========================

st.set_page_config(
    page_title="Tutor IA de Química General EPN",
    page_icon="🧪",
    layout="wide"
)

st.markdown(
    """

<style>
/* ===============================
   PALETA PROFESIONAL CLARA
   Tutor IA de Química General EPN
   =============================== */

:root {
    --bg-main: #F8FAFC;
    --bg-card: #FFFFFF;
    --bg-soft: #EEF6FF;
    --text-main: #0F172A;
    --text-secondary: #334155;
    --text-muted: #64748B;
    --primary: #2563EB;
    --primary-dark: #1E40AF;
    --secondary: #0F766E;
    --secondary-light: #CCFBF1;
    --border: #CBD5E1;
    --shadow: rgba(15, 23, 42, 0.10);
}

/* Fondo general */
.stApp {
    background: linear-gradient(180deg, #F8FAFC 0%, #EFF6FF 100%) !important;
    color: var(--text-main) !important;
}

/* Contenedor principal */
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    max-width: 1120px;
}

/* Encabezado principal */
.hero {
    padding: 1.5rem 1.8rem;
    border-radius: 22px;
    background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 55%, #0F766E 100%);
    border: 1px solid rgba(255,255,255,0.40);
    box-shadow: 0 14px 34px rgba(37, 99, 235, 0.25);
    margin-bottom: 1rem;
}

.hero h1 {
    margin: 0;
    color: #FFFFFF !important;
    font-size: 2.05rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}

.hero p {
    margin-top: 0.45rem;
    color: #E0F2FE !important;
    font-size: 1.02rem;
}

/* Tarjetas generales */
.card {
    padding: 1rem 1.1rem;
    border-radius: 18px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    box-shadow: 0 10px 26px var(--shadow);
    margin-bottom: 0.85rem;
    color: var(--text-main);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: #E5E7EB !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] code {
    color: #0F172A !important;
    background: #E0F2FE !important;
    border-radius: 8px;
    padding: 0.15rem 0.35rem;
}

/* Botones */
.stButton > button {
    background: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12);
}

.stButton > button:hover {
    background: #EFF6FF !important;
    border-color: #2563EB !important;
    color: #1E40AF !important;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
}

.badge-blue {
    background: #DBEAFE;
    color: #1E40AF;
    border: 1px solid #93C5FD;
}

.badge-green {
    background: #CCFBF1;
    color: #0F766E;
    border: 1px solid #5EEAD4;
}

.badge-yellow {
    background: #FEF3C7;
    color: #92400E;
    border: 1px solid #FCD34D;
}

/* Mensajes tipo chat */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 20px !important;
    padding: 1rem !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    color: var(--text-main) !important;
}

/* Texto general */
.stMarkdown,
.stMarkdown p,
.stMarkdown li,
.stMarkdown span,
.stMarkdown div {
    color: var(--text-main) !important;
    font-size: 1.02rem;
    line-height: 1.72;
}

/* Encabezados de respuesta */
.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3 {
    color: #1E40AF !important;
    font-weight: 800 !important;
    letter-spacing: -0.01em;
    margin-top: 1.1rem;
    margin-bottom: 0.55rem;
}

/* Encabezados principales dentro de respuestas */
.stMarkdown h3 {
    font-size: 1.42rem !important;
    padding-bottom: 0.25rem;
    border-bottom: 1px solid #DBEAFE;
}

/* Fórmulas LaTeX */
.katex,
.katex-display {
    color: #0F172A !important;
    font-size: 1.12rem !important;
}

.katex-display {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    margin: 1rem 0;
    overflow-x: auto;
}

/* Cajas de código / pruebas sugeridas */
.stCodeBlock,
pre,
code {
    background: #F1F5F9 !important;
    color: #0F172A !important;
    border-radius: 12px !important;
    border: 1px solid #CBD5E1 !important;
}

/* Input inferior */
[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border-top: 1px solid #CBD5E1 !important;
}

[data-testid="stChatInput"] textarea {
    color: #0F172A !important;
    background: #FFFFFF !important;
    border: 1px solid #94A3B8 !important;
    border-radius: 14px !important;
    font-size: 1rem !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.18) !important;
}

/* Alertas */
.stAlert {
    background: #ECFDF5 !important;
    color: #065F46 !important;
    border: 1px solid #A7F3D0 !important;
    border-radius: 14px !important;
}

/* Expander de fragmentos recuperados */
.streamlit-expanderHeader {
    color: #1E40AF !important;
    font-weight: 700 !important;
    background: #EFF6FF !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 16px !important;
}

/* Separadores */
hr {
    border-color: #CBD5E1 !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #E2E8F0;
}

::-webkit-scrollbar-thumb {
    background: #94A3B8;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #64748B;
}

/* ===============================
   AJUSTE FINAL DE VISIBILIDAD
   =============================== */

/* Fondo principal más limpio */
.stApp {
    background: linear-gradient(180deg, #F8FAFC 0%, #EEF6FF 100%) !important;
}

/* Contenedor principal más ancho y centrado */
.block-container {
    max-width: 1180px !important;
    padding-top: 3.5rem !important;
}

/* Sidebar más profesional */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #60A5FA !important;
    font-weight: 900 !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #E5E7EB !important;
    font-weight: 600 !important;
}

/* Etiquetas del sistema */
[data-testid="stSidebar"] code {
    background: #E0F2FE !important;
    color: #0F172A !important;
    border: 1px solid #7DD3FC !important;
    border-radius: 8px !important;
    padding: 3px 7px !important;
    font-weight: 800 !important;
}

/* Botones de pruebas sugeridas más visibles */
[data-testid="stSidebar"] .stCodeBlock,
[data-testid="stSidebar"] pre,
[data-testid="stSidebar"] code {
    background: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #93C5FD !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
}

/* Botón limpiar */
.stButton > button {
    background: #FFFFFF !important;
    color: #1E3A8A !important;
    border: 1px solid #93C5FD !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
}

.stButton > button:hover {
    background: #DBEAFE !important;
    color: #1E40AF !important;
    border-color: #2563EB !important;
}

/* Mensajes del chat */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.10) !important;
}

/* Texto de respuestas */
.stMarkdown,
.stMarkdown p,
.stMarkdown li,
.stMarkdown span {
    color: #0F172A !important;
    font-size: 1.04rem !important;
    line-height: 1.75 !important;
}

/* Títulos de respuestas */
.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3 {
    color: #1D4ED8 !important;
    font-weight: 900 !important;
}

/* Fórmulas */
.katex,
.katex-display {
    color: #0F172A !important;
}

.katex-display {
    background: #F8FAFC !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 14px !important;
    padding: 12px 16px !important;
}

/* Badges */
.badge-blue {
    background: #DBEAFE !important;
    color: #1E40AF !important;
    border: 1px solid #60A5FA !important;
}

.badge-green {
    background: #D1FAE5 !important;
    color: #047857 !important;
    border: 1px solid #34D399 !important;
}

.badge-yellow {
    background: #FEF3C7 !important;
    color: #92400E !important;
    border: 1px solid #FBBF24 !important;
}

/* Caja de éxito */
.stAlert {
    background: #D1FAE5 !important;
    border: 1px solid #34D399 !important;
    color: #064E3B !important;
    border-radius: 14px !important;
}

/* Entrada inferior */
[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border-top: 1px solid #CBD5E1 !important;
}

[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    color: #0F172A !important;
    border: 2px solid #2563EB !important;
    border-radius: 16px !important;
    font-size: 1rem !important;
}

/* Placeholder del input */
[data-testid="stChatInput"] textarea::placeholder {
    color: #64748B !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 14px !important;
}

.streamlit-expanderHeader {
    color: #1D4ED8 !important;
    font-weight: 800 !important;
}


/* Ajuste de botones de pruebas sugeridas */
[data-testid="stSidebar"] .stCodeBlock,
[data-testid="stSidebar"] pre {
    background: #F8FAFC !important;
    color: #0F172A !important;
    border: 2px solid #60A5FA !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 12px rgba(96, 165, 250, 0.25) !important;
}

[data-testid="stSidebar"] .stCodeBlock code,
[data-testid="stSidebar"] pre code {
    color: #0F172A !important;
    background: transparent !important;
    font-weight: 900 !important;
    font-size: 0.82rem !important;
}

/* Mayor nitidez del título de pruebas sugeridas */
[data-testid="stSidebar"] h3 {
    color: #3B82F6 !important;
    font-size: 1.15rem !important;
    border-bottom: 1px solid #93C5FD !important;
    padding-bottom: 0.3rem !important;
}

/* El chat input más profesional */
[data-testid="stChatInput"] textarea {
    min-height: 46px !important;
    border: 2px solid #2563EB !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.16) !important;
}


/* ===============================
   SIDEBAR FINAL PROFESIONAL
   =============================== */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1220 0%, #111827 55%, #1E293B 100%) !important;
    border-right: 2px solid #1D4ED8 !important;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #60A5FA !important;
    font-weight: 900 !important;
    letter-spacing: -0.01em;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #F8FAFC !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] code {
    background: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #BFDBFE !important;
    border-radius: 10px !important;
    padding: 4px 8px !important;
    font-weight: 900 !important;
}

/* Tarjetas de pruebas sugeridas */
.sidebar-suggestion {
    background: #FFFFFF;
    color: #0F172A !important;
    border: 2px solid #60A5FA;
    border-radius: 14px;
    padding: 12px 13px;
    margin: 11px 0;
    font-size: 0.88rem;
    font-weight: 850;
    line-height: 1.35;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.28);
    word-wrap: break-word;
    overflow-wrap: break-word;
}

.sidebar-suggestion:hover {
    background: #EFF6FF;
    border-color: #2563EB;
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.38);
}

/* Separadores más limpios */
[data-testid="stSidebar"] hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, #93C5FD, transparent) !important;
    margin: 24px 0 !important;
}

/* Botón limpiar conversación */
[data-testid="stSidebar"] .stButton > button {
    background: #F8FAFC !important;
    color: #1E3A8A !important;
    border: 2px solid #60A5FA !important;
    border-radius: 14px !important;
    font-weight: 900 !important;
    padding: 9px 14px !important;
    box-shadow: 0 6px 18px rgba(96, 165, 250, 0.26) !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #DBEAFE !important;
    color: #1D4ED8 !important;
    border-color: #2563EB !important;
}


/* Botón limpiar conversación visible */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    min-height: 42px !important;
    background: #FFFFFF !important;
    color: #0F172A !important;
    border: 2px solid #60A5FA !important;
    border-radius: 14px !important;
    font-weight: 900 !important;
    font-size: 0.88rem !important;
    text-align: center !important;
    box-shadow: 0 6px 18px rgba(96, 165, 250, 0.28) !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #DBEAFE !important;
    color: #1D4ED8 !important;
    border-color: #2563EB !important;
}


/* Botón limpiar conversación negro */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    min-height: 42px !important;
    background: #020617 !important;
    color: #FFFFFF !important;
    border: 2px solid #38BDF8 !important;
    border-radius: 14px !important;
    font-weight: 900 !important;
    font-size: 0.88rem !important;
    text-align: center !important;
    box-shadow: 0 8px 22px rgba(56, 189, 248, 0.35) !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #111827 !important;
    color: #FFFFFF !important;
    border-color: #60A5FA !important;
    box-shadow: 0 10px 26px rgba(96, 165, 250, 0.45) !important;
}


/* ===============================
   BANNER GRANDE DEFINITIVO
   =============================== */

.hero {
    position: relative !important;
    overflow: hidden !important;
    padding: 3rem 3.4rem !important;
    border-radius: 30px !important;
    background:
        radial-gradient(circle at 12% 15%, rgba(45, 212, 191, 0.32), transparent 28%),
        linear-gradient(135deg, #020617 0%, #1E3A8A 42%, #2563EB 68%, #0891B2 100%) !important;
    border: 1px solid rgba(255,255,255,0.32) !important;
    box-shadow: 0 28px 70px rgba(37, 99, 235, 0.36) !important;
    margin: 0 0 28px 0 !important;
    min-height: 190px !important;
}

.hero-inner {
    position: relative !important;
    z-index: 2 !important;
    display: flex !important;
    align-items: center !important;
    gap: 2rem !important;
}

.hero-icon {
    width: 92px !important;
    height: 92px !important;
    min-width: 92px !important;
    border-radius: 26px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 48px !important;
    background: rgba(255,255,255,0.18) !important;
    border: 1px solid rgba(255,255,255,0.38) !important;
    box-shadow: 0 14px 32px rgba(0,0,0,0.26) !important;
}

.hero-kicker {
    color: #A7F3D0 !important;
    font-size: 0.88rem !important;
    font-weight: 900 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    margin-bottom: 0.65rem !important;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 3.35rem !important;
    line-height: 1.02 !important;
    margin: 0 !important;
    padding: 0 !important;
    font-weight: 950 !important;
    letter-spacing: -0.055em !important;
    text-shadow: 0 5px 22px rgba(0,0,0,0.42) !important;
    font-family: "Segoe UI", "Inter", "Arial", sans-serif !important;
}

.hero p {
    color: #E0F2FE !important;
    font-size: 1.12rem !important;
    margin: 1rem 0 0 0 !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    font-family: "Segoe UI", "Inter", "Arial", sans-serif !important;
}


/* ===============================
   BANNER FINAL LIMPIO
   =============================== */

.hero {
    position: relative !important;
    overflow: hidden !important;
    padding: 3rem 3.4rem !important;
    border-radius: 30px !important;
    background:
        radial-gradient(circle at 12% 15%, rgba(45, 212, 191, 0.32), transparent 28%),
        linear-gradient(135deg, #020617 0%, #1E3A8A 42%, #2563EB 68%, #0891B2 100%) !important;
    border: 1px solid rgba(255,255,255,0.32) !important;
    box-shadow: 0 28px 70px rgba(37, 99, 235, 0.36) !important;
    margin: 0 0 28px 0 !important;
    min-height: 190px !important;
}

.hero-inner {
    position: relative !important;
    z-index: 2 !important;
    display: flex !important;
    align-items: center !important;
    gap: 2rem !important;
}

.hero-icon {
    width: 92px !important;
    height: 92px !important;
    min-width: 92px !important;
    border-radius: 26px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 48px !important;
    background: rgba(255,255,255,0.18) !important;
    border: 1px solid rgba(255,255,255,0.38) !important;
    box-shadow: 0 14px 32px rgba(0,0,0,0.26) !important;
}

.hero-kicker {
    color: #A7F3D0 !important;
    font-size: 0.88rem !important;
    font-weight: 900 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    margin-bottom: 0.65rem !important;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 3.35rem !important;
    line-height: 1.02 !important;
    margin: 0 !important;
    padding: 0 !important;
    font-weight: 950 !important;
    letter-spacing: -0.055em !important;
    text-shadow: 0 5px 22px rgba(0,0,0,0.42) !important;
    font-family: "Segoe UI", "Inter", "Arial", sans-serif !important;
}

.hero p {
    color: #E0F2FE !important;
    font-size: 1.12rem !important;
    margin: 1rem 0 0 0 !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    font-family: "Segoe UI", "Inter", "Arial", sans-serif !important;
}


/* ===============================
   BANNER TITULO FINAL SIN H1
   =============================== */

.stMarkdown .hero,
.hero {
    position: relative !important;
    overflow: hidden !important;
    padding: 3rem 3.5rem !important;
    border-radius: 30px !important;
    background:
        radial-gradient(circle at 12% 15%, rgba(45, 212, 191, 0.32), transparent 28%),
        linear-gradient(135deg, #020617 0%, #1E3A8A 42%, #2563EB 68%, #0891B2 100%) !important;
    border: 1px solid rgba(255,255,255,0.32) !important;
    box-shadow: 0 28px 70px rgba(37, 99, 235, 0.36) !important;
    margin: 0 0 28px 0 !important;
    min-height: 190px !important;
}

.stMarkdown .hero-inner,
.hero-inner {
    position: relative !important;
    z-index: 2 !important;
    display: flex !important;
    align-items: center !important;
    gap: 2rem !important;
}

.stMarkdown .hero-icon,
.hero-icon {
    width: 92px !important;
    height: 92px !important;
    min-width: 92px !important;
    border-radius: 26px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 48px !important;
    background: rgba(255,255,255,0.18) !important;
    border: 1px solid rgba(255,255,255,0.38) !important;
    box-shadow: 0 14px 32px rgba(0,0,0,0.26) !important;
}

.stMarkdown .hero-kicker,
.hero-kicker {
    color: #A7F3D0 !important;
    font-size: 0.88rem !important;
    font-weight: 900 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    margin-bottom: 0.75rem !important;
    line-height: 1.2 !important;
}

.stMarkdown .hero-title,
.hero-title {
    color: #FFFFFF !important;
    font-size: 3.35rem !important;
    line-height: 1.02 !important;
    margin: 0 !important;
    padding: 0 !important;
    font-weight: 950 !important;
    letter-spacing: -0.055em !important;
    text-shadow: 0 5px 22px rgba(0,0,0,0.42) !important;
    font-family: "Segoe UI", "Inter", "Arial", sans-serif !important;
}

.stMarkdown .hero-subtitle,
.hero-subtitle {
    color: #E0F2FE !important;
    font-size: 1.12rem !important;
    margin-top: 1rem !important;
    font-weight: 750 !important;
    letter-spacing: -0.01em !important;
    font-family: "Segoe UI", "Inter", "Arial", sans-serif !important;
}



/* ===============================
   SIDEBAR BOTONES COLORES DEFINITIVOS
   =============================== */

/* LIMPIAR CONVERSACIÓN: color naranja/rojo diferente */
[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"],
[data-testid="stSidebar"] div[data-testid="stButton"] [data-testid="baseButton-primary"],
[data-testid="stSidebar"] div[data-testid="stButton"] [data-testid="stBaseButton-primary"] {
    width: 100% !important;
    min-height: 56px !important;
    background: linear-gradient(135deg, #B91C1C 0%, #F97316 55%, #F59E0B 100%) !important;
    color: #FFFFFF !important;
    border: 2px solid #FDBA74 !important;
    border-radius: 18px !important;
    font-weight: 900 !important;
    font-size: 0.98rem !important;
    box-shadow: 0 14px 34px rgba(249, 115, 22, 0.46) !important;
    transition: all 0.22s ease !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] p,
[data-testid="stSidebar"] div[data-testid="stButton"] [data-testid="baseButton-primary"] p,
[data-testid="stSidebar"] div[data-testid="stButton"] [data-testid="stBaseButton-primary"] p {
    color: #FFFFFF !important;
    font-weight: 900 !important;
    font-size: 0.98rem !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"]:hover,
[data-testid="stSidebar"] div[data-testid="stButton"] [data-testid="baseButton-primary"]:hover,
[data-testid="stSidebar"] div[data-testid="stButton"] [data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px) !important;
    background: linear-gradient(135deg, #991B1B 0%, #EA580C 55%, #D97706 100%) !important;
    border-color: #FED7AA !important;
    box-shadow: 0 18px 42px rgba(249, 115, 22, 0.62) !important;
}

/* PRUEBAS SUGERIDAS: azul/celeste distinto */
[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"],
[data-testid="stSidebar"] div[data-testid="stButton"] [data-testid="baseButton-secondary"],
[data-testid="stSidebar"] div[data-testid="stButton"] [data-testid="stBaseButton-secondary"] {
    width: 100% !important;
    min-height: 64px !important;
    height: auto !important;
    background: #020617 !important;
    color: #FFFFFF !important;
    border: 2px solid #38BDF8 !important;
    border-radius: 16px !important;
    padding: 13px 15px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 8px 22px rgba(56, 189, 248, 0.25) !important;
    text-align: center !important;
    white-space: normal !important;
    transition: all 0.22s ease !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] p,
[data-testid="stSidebar"] div[data-testid="stButton"] [data-testid="baseButton-secondary"] p,
[data-testid="stSidebar"] div[data-testid="stButton"] [data-testid="stBaseButton-secondary"] p {
    color: #FFFFFF !important;
    font-weight: 900 !important;
    font-size: 0.88rem !important;
    line-height: 1.35 !important;
    white-space: normal !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"]:hover,
[data-testid="stSidebar"] div[data-testid="stButton"] [data-testid="baseButton-secondary"]:hover,
[data-testid="stSidebar"] div[data-testid="stButton"] [data-testid="stBaseButton-secondary"]:hover {
    transform: translateX(3px) !important;
    background: linear-gradient(135deg, #0F172A 0%, #075985 100%) !important;
    border-color: #7DD3FC !important;
    box-shadow: 0 12px 28px rgba(125, 211, 252, 0.38) !important;
}

</style>

""",
    unsafe_allow_html=True
)


# =========================
# CARGA DE RECURSOS
# =========================

@st.cache_resource(show_spinner=False)
def cargar_recursos():
    skill_text = None
    skill_path_usado = None

    for path in SKILL_PATHS:
        if path.exists():
            skill_text = path.read_text(encoding="utf-8")
            skill_path_usado = str(path)
            break

    if skill_text is None:
        raise FileNotFoundError("No se encontró la skill de Química.")

    modelo_embeddings = SentenceTransformer(EMBEDDING_MODEL_NAME)

    chroma_client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = chroma_client.get_collection(COLLECTION_NAME)

    llm_client = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY
    )

    return {
        "skill": skill_text,
        "skill_path": skill_path_usado,
        "embedding_model": modelo_embeddings,
        "collection": collection,
        "llm_client": llm_client
    }


# =========================
# FUNCIONES DEL TUTOR
# =========================

def preparar_markdown(texto: str) -> str:
    """
    Convierte bloques LaTeX \\[...\\] a $$...$$ para que Streamlit los renderice mejor.
    """
    texto = texto.replace("\\[", "$$")
    texto = texto.replace("\\]", "$$")
    texto = texto.replace("\\(", "$")
    texto = texto.replace("\\)", "$")
    return texto


def parece_comando_terminal(pregunta: str) -> bool:
    comandos = (
        "cd ", "source ", "python ", "pip ", "grep ", "sed ", "nano ",
        "cp ", "mv ", "ls ", "mkdir ", "curl ", "cat ", "rm "
    )
    return pregunta.strip().startswith(comandos)


def clasificar_pregunta(pregunta: str) -> str:
    p = pregunta.lower().strip()

    claves_ejercicio = [
        "calcule", "calcular", "determine", "determina", "balancee",
        "balancear", "convierta", "convertir", "escriba", "halle",
        "obtenga", "resuelva", "masa molar", "estado de oxidación",
        "estado de oxidacion", "configuración electrónica",
        "configuracion electronica", "reactivo limitante", "moles",
        "gramos", "ph", "concentración", "concentracion"
    ]

    claves_conceptual = [
        "qué es", "que es", "explique", "explica", "defina", "define",
        "diferencia", "qué son", "que son", "para qué sirve",
        "para que sirve", "describa", "describe"
    ]

    if any(k in p for k in claves_ejercicio):
        return "ejercicio"

    if any(k in p for k in claves_conceptual):
        return "conceptual"

    if re.search(r"\d", p) or "->" in p or "→" in p:
        return "ejercicio"

    return "conceptual"


def recuperar_fragmentos(pregunta: str, recursos: dict):
    embedding = recursos["embedding_model"].encode(pregunta).tolist()

    resultado = recursos["collection"].query(
        query_embeddings=[embedding],
        n_results=N_FRAGMENTOS,
        include=["documents", "metadatas", "distances"]
    )

    documentos = resultado.get("documents", [[]])[0]
    metadatos = resultado.get("metadatas", [[]])[0]
    distancias = resultado.get("distances", [[]])[0]

    fragmentos = []

    for i, doc in enumerate(documentos):
        meta = metadatos[i] if i < len(metadatos) else {}
        distancia = distancias[i] if i < len(distancias) else None

        fragmentos.append({
            "numero": i + 1,
            "id": meta.get("id", meta.get("chunk_id", "sin_id")),
            "capitulo": meta.get("capitulo", meta.get("chapter", "N/D")),
            "pagina": meta.get("pagina", meta.get("page", "N/D")),
            "tema": meta.get("tema", meta.get("topic", "N/D")),
            "distancia": distancia,
            "texto": doc
        })

    contexto = ""
    for f in fragmentos:
        contexto += f"""
Fragmento {f["numero"]}:
Capítulo: {f["capitulo"]}
Página: {f["pagina"]}
Tema: {f["tema"]}
Contenido:
{f["texto"]}
""".strip() + "\n\n"

    return contexto.strip(), fragmentos


def generar_respuesta_conceptual(pregunta: str, contexto: str, recursos: dict) -> str:
    client = recursos["llm_client"]

    system_prompt = """
Eres un tutor académico especializado en Química General.

La consulta fue clasificada como PREGUNTA CONCEPTUAL.

REGLAS OBLIGATORIAS:
- La respuesta debe iniciar exactamente con: ### Definición
- Está totalmente prohibido usar el formato de ocho apartados.
- No uses encabezados numerados.
- No escribas "1. Fundamento químico del problema".
- No escribas "2. Información proporcionada".
- No escribas "3. Lo que se debe determinar".
- No escribas "4. Relaciones químicas y expresiones necesarias".
- No escribas "5. Aplicación de los datos".
- No escribas "6. Resolución paso a paso".
- No escribas "7. Resultado obtenido".
- No escribas "8. Conclusión".
- No menciones fragmentos, capítulos, páginas, IDs ni distancias.
- No digas "el estudiante pregunta".
- No analices la ortografía de la pregunta.
- No conviertas la pregunta en ejercicio.
- Usa el contexto recuperado solo como apoyo teórico.
- Responde de forma clara, breve, académica y pedagógica.

ESTRUCTURA ÚNICA PERMITIDA:

### Definición

### Explicación

### Ejemplo

### Importancia en Química
""".strip()

    user_prompt = f"""
Pregunta conceptual:
{pregunta}

Contexto teórico de apoyo:
{contexto}

Redacta una respuesta conceptual usando únicamente esta estructura:

### Definición

### Explicación

### Ejemplo

### Importancia en Química

No uses numeración.
No uses los ocho apartados.
No menciones fragmentos recuperados.
""".strip()

    respuesta = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.01,
        max_tokens=650
    )

    texto = respuesta.choices[0].message.content.strip()

    marcas_prohibidas = [
        "### 1.",
        "1. Fundamento químico",
        "Fundamento químico del problema",
        "2. Información proporcionada",
        "Información proporcionada",
        "3. Lo que se debe determinar",
        "Lo que se debe determinar",
        "4. Relaciones químicas",
        "Relaciones químicas y expresiones necesarias",
        "5. Aplicación de los datos",
        "Aplicación de los datos",
        "6. Resolución paso a paso",
        "Resolución paso a paso",
        "7. Resultado obtenido",
        "Resultado obtenido",
        "8. Conclusión",
        "Fragmento",
        "fragmento",
        "Capítulo",
        "capítulo",
        "Página",
        "página",
        "distancia",
        "El estudiante pregunta",
        "Se recuperan"
    ]

    if any(marca in texto for marca in marcas_prohibidas):
        reparacion_prompt = f"""
La siguiente respuesta tiene formato incorrecto porque usa ocho apartados o menciona fragmentos:

{texto}

Reescríbela completamente como una respuesta conceptual a esta pregunta:

{pregunta}

Usa únicamente estos cuatro encabezados:

### Definición

### Explicación

### Ejemplo

### Importancia en Química

No uses numeración.
No uses apartados de ejercicio.
No menciones fragmentos, capítulos ni páginas.
""".strip()

        reparacion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Corrige el formato. Responde solo con cuatro encabezados conceptuales. No uses ocho apartados."
                },
                {
                    "role": "user",
                    "content": reparacion_prompt
                }
            ],
            temperature=0.01,
            max_tokens=550
        )

        texto = reparacion.choices[0].message.content.strip()

    # Último filtro: si aun así insiste en los 8 apartados, se genera una respuesta conceptual mínima.
    if any(marca in texto for marca in marcas_prohibidas):
        texto = f"""### Definición

{pregunta.replace("¿", "").replace("?", "").strip().capitalize()} se refiere a un concepto de Química General que debe explicarse de manera conceptual, sin resolverlo como ejercicio.

### Explicación

En Química, los conceptos se estudian para comprender la composición, estructura, propiedades y transformaciones de la materia. La explicación debe centrarse en el significado del término y en su relación con los fenómenos químicos.

### Ejemplo

Un ejemplo cotidiano permite observar cómo las sustancias poseen propiedades específicas y pueden experimentar cambios físicos o químicos.

### Importancia en Química

Este concepto es importante porque ayuda a comprender el comportamiento de la materia y sirve como base para estudiar sustancias, reacciones, enlaces y propiedades químicas."""
    
    return texto

def generar_respuesta_ejercicio(pregunta: str, contexto: str, recursos: dict) -> str:
    client = recursos["llm_client"]

    system_prompt = f"""
Eres un tutor académico especializado en Química General.

Usa esta skill pedagógica como regla principal:
{recursos["skill"]}

La consulta fue clasificada como EJERCICIO O PROBLEMA DE RESOLUCIÓN.

REGLAS OBLIGATORIAS:
- La respuesta debe iniciar exactamente con: ### 1. Fundamento químico del problema
- Usa exactamente los ocho apartados definidos en la skill.
- No menciones fragmentos recuperados, capítulos, páginas, IDs ni distancias.
- El contexto recuperado es apoyo teórico, no reemplaza los datos del enunciado.
- Si faltan datos indispensables, dilo claramente.
- Si faltan datos, no inventes valores y no presentes un resultado numérico.
- Mantén unidades y verifica cálculos.

CONTROL ESTRICTO DE SECCIONES:
- En `### 2. Información proporcionada`, escribe SOLO datos explícitos del enunciado.
- No adelantes la respuesta final en `### 2. Información proporcionada`.
- No escribas configuraciones electrónicas finales en la sección 2.
- No escribas estados de oxidación finales en la sección 2.
- No escribas ecuaciones balanceadas finales en la sección 2.
- No escribas masas, moles o resultados calculados en la sección 2.
- En `### 5. Aplicación de los datos`, solo sustituye o prepara datos; no presentes el resultado final.
- En `### 6. Resolución paso a paso`, desarrolla el procedimiento.
- En `### 7. Resultado obtenido`, presenta el resultado final.

SI EL EJERCICIO ES DE CONFIGURACIÓN ELECTRÓNICA:
- En la sección 2 coloca únicamente:
  - Elemento químico.
  - Número atómico.
  - Tipo de especie: átomo neutro o ion.
  - Número total de electrones.
- No escribas la configuración electrónica en la sección 2.
- La configuración completa debe aparecer en la sección 6.
- En la sección 7 presenta tanto la configuración completa como la abreviada, si corresponde.
""".strip()

    user_prompt = f"""
Ejercicio del estudiante:
{pregunta}

Contexto recuperado desde la base semántica:
{contexto}

Resuelve el ejercicio con los ocho apartados obligatorios.

Recuerda:
- La sección 2 solo contiene datos del enunciado.
- No adelantes la respuesta en la sección 2.
- El resultado final va únicamente en la sección 7.
""".strip()

    respuesta = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.01,
        max_tokens=1200
    )

    texto = respuesta.choices[0].message.content.strip()

    # Reparación automática si el modelo adelanta la configuración electrónica en sección 2
    pregunta_lower = pregunta.lower()
    es_config = "configuración electrónica" in pregunta_lower or "configuracion electronica" in pregunta_lower

    if es_config:
        seccion2_inicio = texto.find("### 2. Información proporcionada")
        seccion3_inicio = texto.find("### 3. Lo que se debe determinar")

        seccion2 = ""
        if seccion2_inicio != -1 and seccion3_inicio != -1:
            seccion2 = texto[seccion2_inicio:seccion3_inicio]

        adelanta_resultado = (
            "[\\mathrm{Ne}]" in seccion2
            or "[Ne]" in seccion2
            or "1s^2" in seccion2
            or "2p^6" in seccion2
            or "3p^" in seccion2
            or "configuración electrónica del cloro se representa" in seccion2.lower()
        )

        if adelanta_resultado:
            reparacion = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """
Corrige la respuesta de un ejercicio de Química.
Mantén los ocho apartados.
No adelantes resultados en la sección 2.
En la sección 2 solo deben aparecer datos del enunciado.
En configuración electrónica, la configuración final solo puede aparecer en la sección 6 y sección 7.
En la sección 7 coloca configuración completa y abreviada si corresponde.
""".strip()
                    },
                    {
                        "role": "user",
                        "content": f"""
Pregunta original:
{pregunta}

Respuesta incorrecta:
{texto}

Reescribe la respuesta completa corrigiendo la sección 2.
""".strip()
                    }
                ],
                temperature=0.01,
                max_tokens=1200
            )
            texto = reparacion.choices[0].message.content.strip()

    return texto

def generar_respuesta(pregunta: str, recursos: dict):
    modo = clasificar_pregunta(pregunta)
    contexto, fragmentos = recuperar_fragmentos(pregunta, recursos)

    if modo == "conceptual":
        respuesta = generar_respuesta_conceptual(pregunta, contexto, recursos)
    else:
        respuesta = generar_respuesta_ejercicio(pregunta, contexto, recursos)

    return modo, fragmentos, respuesta


def mostrar_fragmentos(fragmentos):
    for f in fragmentos:
        distancia = f["distancia"]
        distancia_txt = f"{distancia:.4f}" if isinstance(distancia, (float, int)) else "N/D"

        st.markdown(
            f"""
<div class="card">
<b>Fragmento {f["numero"]}</b><br>
<span class="badge badge-blue">Capítulo {f["capitulo"]}</span>
<span class="badge badge-green">Página {f["pagina"]}</span>
<span class="badge badge-yellow">Distancia {distancia_txt}</span>
<br><br>
{f["texto"][:650]}...
</div>
""",
            unsafe_allow_html=True
        )


# =========================
# INTERFAZ
# =========================

st.markdown(
    """
<div class="hero">
  <div class="hero-inner">
    <div class="hero-icon">🧪</div>
    <div class="hero-content">
      <div class="hero-kicker">TUTOR ACADÉMICO INTELIGENTE</div>
      <div class="hero-title">Tutor IA de Química General EPN</div>
      <div class="hero-subtitle">Modelo Phi-4 Química · Skill pedagógica · Base semántica de Química General</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("## ⚙️ Sistema")
    st.markdown("**Modelo local:** `phi4-quimica-v2`")
    st.markdown("**Servidor:** `LM Studio`")
    st.markdown("**Base semántica:** `bd_semantica_quimica`")
    st.markdown("**Colección:** `coleccion_libro_quimica_general`")
    st.markdown("**Skill activa:** `tutor-quimica-epn-runtime.md`")

    st.divider()

    if st.button("Limpiar conversación", key="btn_limpiar_chat", type="primary", use_container_width=True):
        st.session_state.mensajes = []
        st.rerun()

    st.divider()
    st.markdown("### 🧪 Pruebas sugeridas")

    sugerencias = [
        "¿Qué es la materia?",
        "Explique la diferencia entre enlace iónico y enlace covalente.",
        "Escriba la configuración electrónica del cloro, cuyo número atómico es 17.",
        "Determine el estado de oxidación del azufre en H₂SO₄.",
    ]

    for i, pregunta_sugerida in enumerate(sugerencias, start=1):
        if st.button(
            pregunta_sugerida,
            key=f"sugerencia_quimica_{i}",
            type="secondary",
            use_container_width=True,
        ):
            st.session_state["pregunta_sugerida"] = pregunta_sugerida
            st.rerun()
try:
    recursos = cargar_recursos()
    st.success(f"Sistema cargado correctamente. Skill: {recursos['skill_path']}")
except Exception as e:
    st.error(f"No se pudieron cargar los recursos: {e}")
    st.stop()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for msg in st.session_state.mensajes:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🧪"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant":
            modo = msg.get("modo", "N/D")
            st.markdown(
                f"""
<span class="badge badge-blue">Modo: {modo}</span>
<span class="badge badge-green">Modelo: {LLM_MODEL}</span>
""",
                unsafe_allow_html=True
            )

        st.markdown(preparar_markdown(msg["content"]))

        if msg["role"] == "assistant" and msg.get("fragmentos"):
            with st.expander("📚 Ver fragmentos recuperados desde la base semántica"):
                mostrar_fragmentos(msg["fragmentos"])

pregunta = st.chat_input("Escribe una pregunta o ejercicio de Química...")

# PROCESAR PREGUNTA SUGERIDA DESDE SIDEBAR
pregunta_sugerida_sidebar = st.session_state.pop("pregunta_sugerida", None)
if pregunta_sugerida_sidebar and not pregunta:
    pregunta = pregunta_sugerida_sidebar

if pregunta:
    pregunta = pregunta.strip()

    st.session_state.mensajes.append({
        "role": "user",
        "content": pregunta
    })

    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(pregunta)

    with st.chat_message("assistant", avatar="🧪"):
        if not pregunta:
            st.warning("Escribe una pregunta válida.")
        elif parece_comando_terminal(pregunta):
            respuesta = "Eso parece un comando de terminal. En esta interfaz escribe solo preguntas de Química."
            st.markdown(respuesta)
            st.session_state.mensajes.append({
                "role": "assistant",
                "content": respuesta,
                "modo": "control",
                "fragmentos": []
            })
        else:
            with st.spinner("Buscando en la base semántica y generando respuesta..."):
                try:
                    modo, fragmentos, respuesta = generar_respuesta(pregunta, recursos)

                    st.markdown(
                        f"""
<span class="badge badge-blue">Modo: {modo}</span>
<span class="badge badge-green">Modelo: {LLM_MODEL}</span>
<span class="badge badge-yellow">RAG activo</span>
""",
                        unsafe_allow_html=True
                    )

                    st.markdown(preparar_markdown(respuesta))

                    with st.expander("📚 Ver fragmentos recuperados desde la base semántica"):
                        mostrar_fragmentos(fragmentos)

                    st.session_state.mensajes.append({
                        "role": "assistant",
                        "content": respuesta,
                        "modo": modo,
                        "fragmentos": fragmentos
                    })

                except Exception as e:
                    error_msg = f"Error al generar la respuesta: {e}"
                    st.error(error_msg)
                    st.session_state.mensajes.append({
                        "role": "assistant",
                        "content": error_msg,
                        "modo": "error",
                        "fragmentos": []
                    })
