import os, re, io
from typing import List, Dict, Tuple
import streamlit as st

st.set_page_config(page_title="Meeting & Email Minutes", layout="wide")
st.title("📝 Meeting & Email Minutes")
st.caption("Paste text or upload .txt / .pdf / .docx → get clean, structured minutes: Decisions, Action Items, Risks, Open Questions, Next Steps.")

# ===================== Sidebar (plain-English hints) =====================
with st.sidebar:
    st.subheader("AI")
    use_ai = st.checkbox(
        "Use AI (OpenAI)",
        value=True,
        help="If ON, the app uses OpenAI to structure minutes. Your API key is read securely from Streamlit 'Secrets'."
    )
    audience = st.selectbox(
        "Audience tone",
        ["Executive", "Operations", "Technical"],
        help="Shapes the writing style (concise for execs, detail for technical)."
    )

    st.markdown("---")
    st.subheader("Presentation")
    polish = st.checkbox(
        "Keep bullets tidy (recommended)",
        value=True,
        help="Cleans small formatting issues like list bullets and spacing."
    )

    st.markdown("---")
    st.subheader("Privacy")
    mask_pii = st.checkbox(
        "Hide personal info before processing",
        value=True,
        help="Replaces emails, phone numbers, and names with tags like [email_1], [phone_1], [name_1] BEFORE any AI call."
    )

# ========================== Small formatting helper ==========================
def normalize_markdown(txt: str) -> str:
    if not txt:
        return ""
    t = txt.strip()
    # unify bullets
    t = re.sub(r'^[\s•*·]\s*', "- ", t, flags=re.MULTILINE)
    # numbered lists "1) " or "1. " -> "1. "
    t = re.sub(r'^\s*(\d+)[\)\.]\s+', r'\1. ', t, flags=re.MULTILINE)
    # blank line after headings like **Decisions**
    t = re.sub(r'(\*\*[^*]+?\*\*)(?!\n\n)', r'\1\n', t)
    # collapse >2 blank lines
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t

# ============================ PII masking helpers ============================
EMAIL_RE = re.compile(r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}', re.I)
PHONE_RE = re.compile(r'(\+?\d[\d\-\s\(\)]{7,}\d)')
NAME_CANDIDATE_RE = re.compile(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})\b')

STOP_TOKENS = {
    "Meeting","Minutes","Action","Actions","Items","Item","Next","Steps",
    "Open","Questions","Decisions","Decision","Project","Policy","Summary",
    "GDPR","AI","Data","LLM","Email","Phone","Owner","Deadline","Risk",
    "Risks","Notes","Agenda","Follow","Up","Follow-Up","Q&A"
}
MONTHS = {"January","February","March","April","May","June","July","August","September",
          "October","November","December","Jan","Feb","Mar","Apr","Jun","Jul","Aug","Sep","Sept","Oct","Nov","Dec"}
DAYS = {"Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday","Mon","Tue","Wed","Thu","Fri","Sat","Sun"}

def looks_like_name(candidate: str) -> bool:
    tokens = candidate.split()
    if len(tokens) < 2 or len(tokens) > 3:
        return False
    for t in tokens:
        if t in STOP_TOKENS or t in MONTHS or t in DAYS:
            return False
        if len(t) <= 2 and t.isupper():
            return False
    return True

def build_pseudonym_subber():
    mapping: Dict[str, str] = {}
    counter = 0
    def sub(match: re.Match) -> str:
        original = match.group(1)
        if not looks_like_name(original):
            return original
        nonlocal counter
        if original not in mapping:
            counter += 1
            mapping[original] = f"[name_{counter}]"
        return mapping[original]
    sub.mapping = mapping  # type: ignore[attr-defined]
    return sub

def mask_text_with_pseudonyms(s: str) -> Tuple[str, Dict[str, str], int, int]:
    email_count = 0
    phone_count = 0
    def sub_email(m: re.Match) -> str:
        nonlocal email_count
        email_count += 1
        return f"[email_{email_count}]"
    def sub_phone(m: re.Match) -> str:
        nonlocal phone_count
        phone_count += 1
        return f"[phone_{phone_count}]"

    masked = EMAIL_RE.sub(sub_email, s)
    masked = PHONE_RE.sub(sub_phone, masked)

    name_subber = build_pseudonym_subber()
    masked = NAME_CANDIDATE_RE.sub(name_subber, masked)

    return masked, getattr(name_subber, "mapping", {}), email_count, phone_count

# ================================ File readers ===============================
def read_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except Exception:
        return file_bytes.decode("latin-1", errors="ignore")

def read_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join((p.extract_text() or "") for p in reader.pages).strip()

def read_docx(file_bytes: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text).strip()

def combine_text(pasted: str, uploaded_texts: List[str]) -> str:
    parts = []
    if pasted and pasted.strip():
        parts.append(pasted.strip())
    for t in uploaded_texts:
        if t and t.strip():
            parts.append(t.strip())
    return "\n\n---\n\n".join(parts)

# ============================= OpenAI caller (raw errors) ====================
def call_openai_minutes(prompt: str) -> str:
    """
    Uses your OPENAI_API_KEY from Streamlit Secrets or env.
    Tries new SDK first; if import/type errors, falls back to legacy.
    On error, raises the original exception so the UI can display it verbatim.
    """
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit Secrets or environment.")

    # Try new SDK
    try:
        from openai import OpenAI  # new SDK
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=700
        )
        return resp.choices[0].message.content.strip()
    except (TypeError, AttributeError):
        # Fall back to legacy SDK signature
        pass

    # Legacy fallback
    import openai as openai_legacy
    openai_legacy.api_key = api_key
    resp = openai_legacy.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=700,
    )
    return resp["choices"][0]["message"]["content"].strip()

# ================================ Prompt builder =============================
def build_prompt(audience: str, raw: str) -> str:
    return f"""You are a precise note-taker. Audience: {audience}.
Structure the following text into concise minutes.

Sections:
- **Decisions**
- **Action Items** (Owner, Deadline)
- **Risks**
- **Open Questions**
- **Next Steps** (short bullet list)

Rules:
- No fluff. Use bullet points.
- If owners/dates are missing, infer placeholders and mark with (?).
- Keep it business-friendly.

Text:
{raw}
"""

# ================================== Inputs ==================================
c1, c2 = st.columns([1,1])
with c1:
    raw = st.text_area(
        "Paste meeting transcript or email thread",
        height=260,
        placeholder="Paste here…",
        help="Tip: You can paste raw meeting notes or an email chain. The app will structure it for quick sharing."
    )
with c2:
    files = st.file_uploader(
        "Upload files (.txt, .pdf, .docx) — optional",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        help="If you have notes as files, drop them here. Scanned PDFs may extract poorly."
    )

uploaded_texts: List[str] = []
if files:
    for f in files:
        data = f.read()
        if f.type == "text/plain" or f.name.lower().endswith(".txt"):
            uploaded_texts.append(read_txt(data))
        elif f.name.lower().endswith(".pdf"):
            uploaded_texts.append(read_pdf(data))
        elif f.name.lower().endswith(".docx"):
            uploaded_texts.append(read_docx(data))

full_text = combine_text(raw, uploaded_texts)

with st.expander("Preview extracted text (first 5,000 chars)", expanded=False):
    st.write(full_text[:5000] if full_text else "—")

st.markdown(
    "> **Privacy note:** Nothing is stored. If AI is ON, text is sent to OpenAI **after optional masking**. "
    "If AI is OFF, the app shows a clean minutes template you can edit."
)

# ================================== Action ==================================
if st.button("Generate Minutes"):
    if not full_text.strip():
        st.warning("Please paste text or upload at least one file.")
        st.stop()

    # Mask BEFORE any AI usage (if enabled)
    if mask_pii:
        processed_text, name_map, email_cnt, phone_cnt = mask_text_with_pseudonyms(full_text)
    else:
        processed_text, name_map, email_cnt, phone_cnt = full_text, {}, 0, 0

    if use_ai:
        try:
            prompt = build_prompt(audience, processed_text)
            raw_out = call_openai_minutes(prompt)
        except Exception as e:
            # Show the exact error from OpenAI (rate limit, quota, etc.)
            st.error(f"OpenAI error: {e}")
            st.stop()
    else:
        # No AI: simple editable template (no heuristics)
        raw_out = (
            "**Decisions**\n- —\n\n"
            "**Action Items (Owner, Deadline)**\n- —\n\n"
            "**Risks**\n- —\n\n"
            "**Open Questions**\n- —\n\n"
            "**Next Steps**\n- —"
        )

    output = normalize_markdown(raw_out) if polish else raw_out
    st.subheader("Minutes")
    st.markdown(output)

    with st.expander("Masking summary (no raw PII shown)", expanded=False):
        st.write(f"- Emails masked: {email_cnt}")
        st.write(f"- Phones masked: {phone_cnt}")
        if name_map:
            st.write(f"- Names masked: {len(name_map)} → {[name_map[k] for k in name_map]}")
        else:
            st.write("- Names masked: 0")
