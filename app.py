import os, re, io
from typing import List, Dict, Tuple
import streamlit as st

st.set_page_config(page_title="Meeting & Email Minutes", layout="wide")
st.title("📝 Meeting & Email Minutes")
st.caption("Paste text or upload .txt / .pdf / .docx → get clean, structured minutes: Decisions, Action Items, Risks, Open Questions, Next Steps.")

# ===================== Sidebar (simple) =====================
with st.sidebar:
    st.subheader("AI")
    st.write("Model: **gpt-4o-mini**")
    audience = st.selectbox(
        "Audience tone",
        ["Executive", "Operations", "Technical"],
        help="Shapes the writing style (concise for execs, process for ops, detailed for technical)."
    )
    st.markdown("---")
    polish = st.checkbox(
        "Keep bullets tidy (recommended)",
        value=True,
        help="Cleans spacing and guarantees bullets only under headings — never on headings."
    )
    st.markdown("---")
    mask_pii = st.checkbox(
        "Hide personal info before processing",
        value=True,
        help="Replaces emails, phone numbers, and names with tags like [email_1], [phone_1], [name_1] BEFORE any AI call."
    )

# ============================ PII masking ============================
EMAIL_RE = re.compile(r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}', re.I)
PHONE_RE = re.compile(r'(\+?\d[\d\-\s\(\)]{7,}\d)')
NAME_CANDIDATE_RE = re.compile(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})\b')

STOP_TOKENS = {"Meeting","Minutes","Action","Actions","Items","Item","Next","Steps",
               "Open","Questions","Decisions","Decision","Project","Policy","Summary",
               "GDPR","AI","Data","LLM","Email","Phone","Owner","Deadline","Risk",
               "Risks","Notes","Agenda","Follow","Up","Follow-Up","Q&A"}
MONTHS = {"January","February","March","April","May","June","July","August","September",
          "October","November","December","Jan","Feb","Mar","Apr","Jun","Jul","Aug","Sep","Sept","Oct","Nov","Dec"}
DAYS = {"Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday",
        "Mon","Tue","Wed","Thu","Fri","Sat","Sun"}

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

# ============================= Prompt (audience-specific style) ========================
def build_prompt(audience: str, raw: str) -> str:
    if audience == "Executive":
        style = (
            "Write in concise, high-level language. "
            "Focus on key business decisions, commitments, and deadlines. "
            "Avoid technical jargon or minor details."
        )
    elif audience == "Technical":
        style = (
            "Write with detailed explanations. "
            "Include technical tasks, system dependencies, and implementation notes where present. "
            "Use domain-specific terminology if found in the text."
        )
    else:  # Operations
        style = (
            "Write in a process- and logistics-focused style. "
            "Emphasize risks, dependencies, handoffs, and schedules. "
            "Make action steps and responsibilities very explicit."
        )

    return f"""You are a precise note-taker. Audience: {audience}.
{style}

Rewrite the text into business minutes using EXACTLY this Markdown layout — no extra text above or below:

### Decisions
- item

### Action Items
- Owner: <name or [name_1]> — Task … (Deadline: <date or (?)>)

### Risks
- item

### Open Questions
- item

### Next Steps
- item

Rules:
- Use headings exactly as written above.
- Bullets only under headings, never on headings.
- Keep it {audience.lower()}-friendly.
- If the owner or deadline is unclear, write '(?)'.
- Do not include “Minutes”, “Summary”, or any other heading.
- Do not add any other sections.

Text to structure:
{raw}
"""

# ====================== Post-processor (canonical layout) ====================
HEADINGS = ["Decisions", "Action Items", "Risks", "Open Questions", "Next Steps"]

def canonicalize_minutes(md: str) -> str:
    """
    Convert any model quirks into the exact heading + bullets shape:
    - Normalize headings to '### <Heading>'
    - Never bullet the headings
    - Bullet only lines under a known section
    - Strip stray '*' around headings
    """
    lines = [ln.rstrip() for ln in md.strip().splitlines()]
    out = []
    current = None

    heading_pat = re.compile(r'^\s*[*_#\-\s]*\s*(Decisions|Action Items|Risks|Open Questions|Next Steps)\s*:?\s*[*_#\s]*$', re.I)
    item_lead_pat = re.compile(r'^\s*[-•*·]\s*')

    def emit_heading(h: str):
        out.append(f"### {h}")
        out.append("")

    for raw in lines:
        txt = raw.strip()
        if not txt:
            continue

        m = heading_pat.match(txt)
        if m:
            current = m.group(1).title()
            if current in HEADINGS:
                emit_heading(current)
            else:
                current = None
            continue

        star_stripped = re.sub(r'^\*+|\*+$', '', txt).strip()
        m2 = heading_pat.match(star_stripped)
        if m2:
            current = m2.group(1).title()
            if current in HEADINGS:
                emit_heading(current)
            else:
                current = None
            continue

        if current is None:
            continue
        else:
            item = item_lead_pat.sub("", txt)
            if item and not item.startswith("###"):
                out.append(f"- {item}")

    present = set([ln[4:] for ln in out if ln.startswith("### ")])
    for h in HEADINGS:
        if h not in present:
            if out and out[-1] != "":
                out.append("")
            out.append(f"### {h}")
            out.append("")
            out.append("- —")

    final = []
    prev_blank = False
    for ln in out:
        if ln == "":
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False
        final.append(ln)
    return "\n".join(final).strip() + "\n"

# ============================= OpenAI (legacy SDK 0.28.x) ===================
def call_openai_minutes(prompt: str) -> str:
    """
    Uses OpenAI Python SDK 0.28.x (ChatCompletion).
    Reads OPENAI_API_KEY from Streamlit Secrets or env. Shows raw errors.
    """
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit Secrets or environment.")

    import openai
    openai.api_key = api_key

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800,
    )
    return resp["choices"][0]["message"]["content"].strip()

# ================================== Inputs ==================================
c1, c2 = st.columns([1,1])
with c1:
    raw = st.text_area(
        "Paste meeting transcript or email thread",
        height=260,
        placeholder="Paste here…",
        help="Tip: Paste raw meeting notes or an email chain. The app will structure it for quick sharing."
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
    "> **Privacy note:** Nothing is stored. If AI is ON, text is sent to OpenAI **after optional masking**."
)

# ================================== Action (with BIG progress box) ==========
generate = st.button("Generate Minutes", type="primary")
if generate:
    if not full_text.strip():
        st.warning("Please paste text or upload at least one file.")
        st.stop()

    with st.status("Generating minutes…", expanded=True) as status:
        status.update(label="Masking personal info…")
        if mask_pii:
            processed_text, name_map, email_cnt, phone_cnt = mask_text_with_pseudonyms(full_text)
        else:
            processed_text, name_map, email_cnt, phone_cnt = full_text, {}, 0, 0

        status.update(label="Calling OpenAI…")
        try:
            prompt = build_prompt(audience, processed_text)
            raw_out = call_openai_minutes(prompt)
        except Exception as e:
            status.update(label="OpenAI error — showing editable template.", state="error")
            st.error(f"OpenAI error: {e}")
            raw_out = (
                "### Decisions\n- —\n\n"
                "### Action Items\n- —\n\n"
                "### Risks\n- —\n\n"
                "### Open Questions\n- —\n\n"
                "### Next Steps\n- —\n"
            )
        else:
            status.update(label="Formatting output…")

        rendered = canonicalize_minutes(raw_out)
        if polish:
            rendered = re.sub(r'\n{3,}', '\n\n', rendered).strip() + "\n"

        status.update(label="Done ✅", state="complete")

    st.subheader("Minutes")
    st.markdown(rendered)

    with st.expander("Masking summary (no raw PII shown)", expanded=False):
        st.write(f"- Emails masked: {email_cnt}")
        st.write(f"- Phones masked: {phone_cnt}")
        if name_map:
            st.write(f"- Names masked: {len(name_map)} → {[name_map[k] for k in name_map]}")
        else:
            st.write("- Names masked: 0")
