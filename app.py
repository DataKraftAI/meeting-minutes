import os, re, io
from typing import List, Dict, Tuple
import streamlit as st

# -------- Optional OpenAI (the app still works without it) --------
try:
    from openai import OpenAI
    OPENAI_READY = True
except Exception:
    OPENAI_READY = False

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

    st.markdown("---")
    st.subheader("No-AI Extraction")
    local_heuristics = st.checkbox(
        "If AI is OFF, extract with simple rules",
        value=True,
        help="Pure Python rules (no external services) to pull obvious Decisions, Action Items, Risks, etc."
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

# ========================== Local (no-AI) extractor ==========================
DATE_PAT = re.compile(
    r'\b('
    r'\d{4}-\d{2}-\d{2}'                      # 2025-10-06
    r'|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'         # 06/10/2025 or 6-10-25
    r'|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.? \d{1,2},? \d{2,4}'  # Oct 6, 2025
    r'|\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.? \d{2,4}'    # 6 Oct 2025
    r')\b',
    re.I
)

OWNER_PAT = re.compile(r'\[(name_\d+)\]', re.I)  # after masking
CAP_OWNER_PAT = re.compile(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})\b')  # if masking off

ACTION_VERBS = r'(?:will|shall|must|to|own|draft|update|finalize|coordinate|prepare|send|review|implement|fix|deliver|publish|clarify)'
DEADLINE_HINTS = r'(?:by|before|on|due|no later than|target (?:date|deadline) is|target is|goal is|target:|deadline:)'

def find_deadline(text: str) -> str:
    # priority: explicit date near deadline hints; else any date in sentence; else "?"
    hint = re.search(rf'{DEADLINE_HINTS}[^\.:\n]*', text, re.I)
    if hint:
        m = DATE_PAT.search(hint.group(0))
        if m:
            return m.group(0)
    m2 = DATE_PAT.search(text)
    return m2.group(0) if m2 else "(?)"

def sentencize(text: str) -> List[str]:
    # split by line breaks and sentence punctuation, keep medium granularity
    chunks = []
    for block in text.splitlines():
        block = block.strip()
        if not block:
            continue
        parts = re.split(r'(?<=[\.\!\?])\s+(?=[A-Z\[])', block)
        for p in parts:
            p = p.strip()
            if p:
                chunks.append(p)
    return chunks

def local_extract(text: str, masking_active: bool) -> Dict[str, List[str]]:
    sections = {
        "Decisions": [],
        "Action Items (Owner, Deadline)": [],
        "Risks": [],
        "Open Questions": [],
        "Next Steps": []
    }

    # Heading-based buckets (e.g., "Risks:", "Next steps:", "Open questions:")
    lines = [ln.strip() for ln in text.splitlines()]
    current = None
    for ln in lines:
        low = ln.lower().strip()
        if not low:
            current = None
            continue
        if low.startswith(("risks:", "risk:")):
            current = "Risks"; continue
        if low.startswith(("next steps:", "next step:", "follow-up:", "follow up:")):
            current = "Next Steps"; continue
        if low.startswith(("open questions:", "open question:", "questions:")):
            current = "Open Questions"; continue
        if low.startswith(("decisions:", "decision:")):
            current = "Decisions"; continue

        if current in ("Risks","Next Steps","Open Questions","Decisions"):
            # Avoid echoing the heading itself
            if ln and not re.match(r'^(risks?|next steps?|follow-?up|open questions?|questions?):\s*$', ln, re.I):
                bullet = f"- {ln}" if not ln.startswith("-") else ln
                sections[current].append(bullet)

    # Sentence-level rules
    for s in sentencize(text):
        low = s.lower()

        # Decisions
        if "we agreed" in low or "we decided" in low or low.startswith("decision:"):
            sections["Decisions"].append(f"- {s}")
            continue

        # Risks (keywords)
        if "risk" in low or "blocked" in low or "blocker" in low or "jeopardize" in low or "dependency" in low or "dependent on" in low:
            sections["Risks"].append(f"- {s}")
            continue

        # Next steps (phrases)
        if "next steps" in low or "let’s reconvene" in low or "let's reconvene" in low or "follow up" in low:
            sections["Next Steps"].append(f"- {s}")
            continue

        # Open questions (question mark)
        if s.endswith("?"):
            sections["Open Questions"].append(f"- {s}")
            continue

        # Action items:
        # 1) Masked owner like [name_1] ... will/shall/to/etc.
        if re.search(rf'\[name_\d+\].{{0,40}}\b{ACTION_VERBS}\b', low):
            owner_match = OWNER_PAT.search(s)
            owner = f'[{owner_match.group(1)}]' if owner_match else "[name_(?)]"
            deadline = find_deadline(s)
            sections["Action Items (Owner, Deadline)"].append(f"- {s} (Owner: {owner}, Deadline: {deadline})")
            continue

        # 2) If masking is OFF, try capitalized names as owners
        if not masking_active:
            # crude owner guess: sentence starts with Name ... verb
            m = re.match(rf'({CAP_OWNER_PAT.pattern}).{{0,40}}\b{ACTION_VERBS}\b', s)
            if m:
                owner = m.group(1)
                deadline = find_deadline(s)
                sections["Action Items (Owner, Deadline)"].append(f"- {s} (Owner: {owner}, Deadline: {deadline})")
                continue

    return sections

def render_sections(sections: Dict[str, List[str]]) -> str:
    def block(title: str, items: List[str]) -> str:
        if not items:
            return f"**{title}**\n- —\n"
        # de-duplicate while preserving order
        seen = set()
        cleaned = []
        for it in items:
            if it not in seen:
                cleaned.append(it)
                seen.add(it)
        return f"**{title}**\n" + "\n".join(cleaned) + "\n"
    return (
        block("Decisions", sections.get("Decisions", [])) + "\n" +
        block("Action Items (Owner, Deadline)", sections.get("Action Items (Owner, Deadline)", [])) + "\n" +
        block("Risks", sections.get("Risks", [])) + "\n" +
        block("Open Questions", sections.get("Open Questions", [])) + "\n" +
        block("Next Steps", sections.get("Next Steps", []))
    )

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
    "If AI is OFF, everything stays in this session."
)

# ================================== Action ==================================
if st.button("Generate Minutes"):
    if not full_text.strip():
        st.warning("Please paste text or upload at least one file.")
        st.stop()

    # Mask BEFORE any AI usage (if enabled)
    if mask_pii:
        processed_text, name_map, email_cnt, phone_cnt = mask_text_with_pseudonyms(full_text)
        masking_active = True
    else:
        processed_text, name_map, email_cnt, phone_cnt = full_text, {}, 0, 0
        masking_active = False

    if use_ai:
        # Read key from Streamlit Secrets or env — no user input
        api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        if not OPENAI_READY or not api_key:
            st.error("This demo is temporarily unavailable (missing OpenAI key). Please try again later.")
            st.stop()

        try:
            client = OpenAI(api_key=api_key)
            prompt = build_prompt(audience, processed_text)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=700
            )
            raw_out = resp.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ["quota", "insufficient", "rate", "exceeded"]):
                st.warning("⚠️ Demo limit exceeded for this month. Please check back next month.")
            else:
                st.error(f"OpenAI error: {e}")
            st.stop()
    else:
        # No-AI mode: rule-based extraction or simple template
        if local_heuristics:
            sections = local_extract(processed_text, masking_active=masking_active)
            raw_out = render_sections(sections)
        else:
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
