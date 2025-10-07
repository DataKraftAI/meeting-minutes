import os, re, io
from typing import List, Dict, Tuple
import streamlit as st

st.set_page_config(page_title="Meeting & Email Minutes / Besprechungsprotokoll", layout="wide")

# ------------------------ Language bootstrap ------------------------
def get_default_lang() -> str:
    # Prefer URL param: ?lang=de or ?lang=en
    try:
        params = st.experimental_get_query_params()  # works on Streamlit Cloud
        lang = (params.get("lang", ["en"])[0] or "en").lower()
        return "de" if lang.startswith("de") else "en"
    except Exception:
        return "en"

LANG = st.session_state.get("lang", get_default_lang())
if "lang" not in st.session_state:
    st.session_state["lang"] = LANG

def set_lang(new_lang: str):
    st.session_state["lang"] = new_lang

# ------------------------ Localization strings ------------------------
TXT = {
    "en": {
        "title": "📝 Meeting & Email Minutes",
        "caption": "Paste text or upload .txt / .pdf / .docx → get clean, structured minutes: Decisions, Action Items, Risks, Open Questions, Next Steps.",
        "sidebar_ai": "AI",
        "model": "Model: **gpt-4o-mini**",
        "audience_label": "Audience tone",
        "audience_help": "Shapes the writing style (concise for execs, process for ops, detailed for technical).",
        "aud_exec": "Executive",
        "aud_ops": "Operations",
        "aud_tech": "Technical",
        "presentation": "Presentation",
        "polish_label": "Keep bullets tidy (recommended)",
        "polish_help": "Cleans spacing and guarantees bullets only under headings — never on headings.",
        "privacy": "Privacy",
        "mask_label": "Hide personal info before processing",
        "mask_help": "Replaces emails, phone numbers, and names with tags like [email_1], [phone_1], [name_1] BEFORE any AI call.",
        "paste_label": "Paste meeting transcript or email thread",
        "paste_help": "Tip: Paste raw meeting notes or an email chain. The app will structure it for quick sharing.",
        "upload_label": "Upload files (.txt, .pdf, .docx) — optional",
        "upload_help": "If you have notes as files, drop them here. Scanned PDFs may extract poorly.",
        "preview": "Preview extracted text (first 5,000 chars)",
        "privacy_note": "> **Privacy note:** Nothing is stored. If AI is ON, text is sent to OpenAI **after optional masking**.",
        "btn_generate": "Generate Minutes",
        "status_title": "Generating minutes…",
        "status_mask": "Masking personal info…",
        "status_call": "Calling OpenAI…",
        "status_format": "Formatting output…",
        "status_done": "Done ✅",
        "warn_paste": "Please paste text or upload at least one file.",
        "error_openai_prefix": "OpenAI error: ",
        "minutes_heading": "Minutes",
        "mask_summary": "Masking summary (no raw PII shown)",
        # Headings
        "H_DECISIONS": "Decisions",
        "H_ACTIONS": "Action Items",
        "H_RISKS": "Risks",
        "H_QUESTIONS": "Open Questions",
        "H_NEXT": "Next Steps",
    },
    "de": {
        "title": "📝 Besprechungsprotokoll",
        "caption": "Text einfügen oder .txt / .pdf / .docx hochladen → klare, strukturierte Protokolle: Entscheidungen, Aufgaben, Risiken, Offene Fragen, Nächste Schritte.",
        "sidebar_ai": "KI",
        "model": "Modell: **gpt-4o-mini**",
        "audience_label": "Adressatenton",
        "audience_help": "Beeinflusst den Stil (kurz für Management, prozessorientiert für Betrieb, detailliert für Technik).",
        "aud_exec": "Management",
        "aud_ops": "Betrieb",
        "aud_tech": "Technik",
        "presentation": "Darstellung",
        "polish_label": "Aufzählungen sauber halten (empfohlen)",
        "polish_help": "Sorgt für saubere Abstände und Aufzählungen nur unter Überschriften — nie bei Überschriften.",
        "privacy": "Datenschutz",
        "mask_label": "Personenbezogene Daten vor der Verarbeitung ausblenden",
        "mask_help": "Ersetzt E-Mails, Telefonnummern und Namen durch Tags wie [email_1], [phone_1], [name_1], BEVOR etwas an die KI gesendet wird.",
        "paste_label": "Besprechungsnotizen oder E-Mail-Verlauf einfügen",
        "paste_help": "Tipp: Fügen Sie Rohnotizen oder eine E-Mail-Kette ein. Die App strukturiert alles für die schnelle Weitergabe.",
        "upload_label": "Dateien hochladen (.txt, .pdf, .docx) — optional",
        "upload_help": "Wenn Notizen als Dateien vorliegen, hier ablegen. Gescannte PDFs werden ggf. schlecht extrahiert.",
        "preview": "Vorschau des extrahierten Textes (erste 5.000 Zeichen)",
        "privacy_note": "> **Hinweis Datenschutz:** Es wird nichts gespeichert. Wenn KI aktiviert ist, wird Text **nach optionaler Anonymisierung** an OpenAI gesendet.",
        "btn_generate": "Protokoll erstellen",
        "status_title": "Protokoll wird erstellt…",
        "status_mask": "Personenbezogene Daten werden anonymisiert…",
        "status_call": "OpenAI wird aufgerufen…",
        "status_format": "Ausgabe wird formatiert…",
        "status_done": "Fertig ✅",
        "warn_paste": "Bitte Text einfügen oder mindestens eine Datei hochladen.",
        "error_openai_prefix": "OpenAI-Fehler: ",
        "minutes_heading": "Protokoll",
        "mask_summary": "Zusammenfassung der Anonymisierung (keine Roh-PII)",
        # Headings
        "H_DECISIONS": "Entscheidungen",
        "H_ACTIONS": "Aufgaben",
        "H_RISKS": "Risiken",
        "H_QUESTIONS": "Offene Fragen",
        "H_NEXT": "Nächste Schritte",
    }
}

# Title + language selector row
col_t, col_lang = st.columns([1, 0.28])
with col_t:
    st.title(TXT[LANG]["title"])
    st.caption(TXT[LANG]["caption"])
with col_lang:
    new_lang = st.selectbox("Language / Sprache", ["English", "Deutsch"], index=(0 if LANG=="en" else 1))
    picked = "en" if new_lang.startswith("English") else "de"
    if picked != LANG:
        set_lang(picked)
        st.experimental_set_query_params(lang=picked)
        st.experimental_rerun()

# ===================== Sidebar (simple) =====================
with st.sidebar:
    st.subheader(TXT[LANG]["sidebar_ai"])
    st.write(TXT[LANG]["model"])
    audience_opts = [TXT[LANG]["aud_exec"], TXT[LANG]["aud_ops"], TXT[LANG]["aud_tech"]]
    audience = st.selectbox(TXT[LANG]["audience_label"], audience_opts, help=TXT[LANG]["audience_help"])
    st.markdown("---")
    polish = st.checkbox(TXT[LANG]["polish_label"], value=True, help=TXT[LANG]["polish_help"])
    st.markdown("---")
    mask_pii = st.checkbox(TXT[LANG]["mask_label"], value=True, help=TXT[LANG]["mask_help"])

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

# =========== Audience-specific prompt builder (EN + DE) ===========
def build_prompt(audience_label: str, raw: str, lang: str) -> str:
    if lang == "de":
        # Map localized audience labels back to styles
        if audience_label == TXT["de"]["aud_exec"]:
            style = ("Schreiben Sie kurz und auf Management-Ebene. "
                     "Fokus auf Entscheidungen, Zusagen und Termine. "
                     "Vermeiden Sie technischen Jargon oder Nebensächlichkeiten.")
        elif audience_label == TXT["de"]["aud_tech"]:
            style = ("Schreiben Sie mit detaillierten Erklärungen. "
                     "Beziehen Sie technische Aufgaben, System-Abhängigkeiten und Implementierungs-Hinweise ein, sofern vorhanden. "
                     "Nutzen Sie Fachbegriffe, falls sie im Text vorkommen.")
        else:  # Betrieb
            style = ("Schreiben Sie prozess- und logistikorientiert. "
                     "Betonen Sie Risiken, Abhängigkeiten, Übergaben und Zeitpläne. "
                     "Machen Sie Aufgaben und Verantwortlichkeiten sehr deutlich.")
        return f"""Sie sind ein präziser Protokollführer. Zielgruppe: {audience_label}.
{style}

Strukturieren Sie den Text in folgendes Markdown-Layout — kein zusätzlicher Text davor oder danach:

### {TXT['de']['H_DECISIONS']}
- punkt

### {TXT['de']['H_ACTIONS']}
- Verantwortlich: <Name oder [name_1]> — Aufgabe … (Termin: <Datum oder (?)>)

### {TXT['de']['H_RISKS']}
- punkt

### {TXT['de']['H_QUESTIONS']}
- punkt

### {TXT['de']['H_NEXT']}
- punkt

Regeln:
- Verwenden Sie die Überschriften exakt wie oben (ohne Fettdruck/Kursiv, keine Doppelpunkte).
- Unter jeder Überschrift nur '-' Aufzählungen, nicht bei Überschriften.
- Wenn Verantwortliche oder Termin unklar sind, '(?)' schreiben.
- Keine weiteren Abschnitte hinzufügen.

Zu strukturierender Text:
{raw}
"""
    else:
        # English
        if audience_label == TXT["en"]["aud_exec"]:
            style = ("Write in concise, high-level language. "
                     "Focus on key business decisions, commitments, and deadlines. "
                     "Avoid technical jargon or minor details.")
        elif audience_label == TXT["en"]["aud_tech"]:
            style = ("Write with detailed explanations. "
                     "Include technical tasks, system dependencies, and implementation notes where present. "
                     "Use domain-specific terminology if found in the text.")
        else:  # Operations
            style = ("Write in a process- and logistics-focused style. "
                     "Emphasize risks, dependencies, handoffs, and schedules. "
                     "Make action steps and responsibilities very explicit.")
        return f"""You are a precise note-taker. Audience: {audience_label}.
{style}

Rewrite the text into business minutes using EXACTLY this Markdown layout — no extra text above or below:

### {TXT['en']['H_DECISIONS']}
- item

### {TXT['en']['H_ACTIONS']}
- Owner: <name or [name_1]> — Task … (Deadline: <date or (?)>)

### {TXT['en']['H_RISKS']}
- item

### {TXT['en']['H_QUESTIONS']}
- item

### {TXT['en']['H_NEXT']}
- item

Rules:
- Use headings exactly as written above.
- Bullets only under headings, never on headings.
- If the owner or deadline is unclear, write '(?)'.
- Do not add any other sections.

Text to structure:
{raw}
"""

# ====================== Post-processor (canonical layout) ====================
def canonicalize_minutes(md: str, lang: str) -> str:
    if lang == "de":
        H = [TXT["de"]["H_DECISIONS"], TXT["de"]["H_ACTIONS"], TXT["de"]["H_RISKS"], TXT["de"]["H_QUESTIONS"], TXT["de"]["H_NEXT"]]
    else:
        H = [TXT["en"]["H_DECISIONS"], TXT["en"]["H_ACTIONS"], TXT["en"]["H_RISKS"], TXT["en"]["H_QUESTIONS"], TXT["en"]["H_NEXT"]]

    lines = [ln.rstrip() for ln in md.strip().splitlines()]
    out = []
    current = None

    heading_pat = re.compile(r'^\s*[*_#\-\s]*\s*(' + "|".join(map(re.escape, H)) + r')\s*:?\s*[*_#\s]*$', re.I)
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
            current = m.group(1)
            if current in H:
                emit_heading(current)
            else:
                current = None
            continue

        star_stripped = re.sub(r'^\*+|\*+$', '', txt).strip()
        m2 = heading_pat.match(star_stripped)
        if m2:
            current = m2.group(1)
            if current in H:
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
    for h in H:
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
        TXT[LANG]["paste_label"],
        height=260,
        placeholder="Paste here…" if LANG=="en" else "Hier einfügen…",
        help=TXT[LANG]["paste_help"]
    )
with c2:
    files = st.file_uploader(
        TXT[LANG]["upload_label"],
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        help=TXT[LANG]["upload_help"]
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

with st.expander(TXT[LANG]["preview"], expanded=False):
    st.write(full_text[:5000] if full_text else "—")

st.markdown(TXT[LANG]["privacy_note"])

# ================================== Action (with BIG progress box) ==========
generate = st.button(TXT[LANG]["btn_generate"], type="primary")
if generate:
    if not full_text.strip():
        st.warning(TXT[LANG]["warn_paste"])
        st.stop()

    with st.status(TXT[LANG]["status_title"], expanded=True) as status:
        status.update(label=TXT[LANG]["status_mask"])
        if mask_pii:
            processed_text, name_map, email_cnt, phone_cnt = mask_text_with_pseudonyms(full_text)
        else:
            processed_text, name_map, email_cnt, phone_cnt = full_text, {}, 0, 0

        status.update(label=TXT[LANG]["status_call"])
        try:
            prompt = build_prompt(audience, processed_text, LANG)
            raw_out = call_openai_minutes(prompt)
        except Exception as e:
            status.update(label="OpenAI error — showing editable template." if LANG=="en" else "OpenAI-Fehler — bearbeitbare Vorlage wird angezeigt.", state="error")
            st.error(TXT[LANG]["error_openai_prefix"] + str(e))
            # Fallback template in correct language
            raw_out = (
                f"### {TXT[LANG]['H_DECISIONS']}\n- —\n\n"
                f"### {TXT[LANG]['H_ACTIONS']}\n- —\n\n"
                f"### {TXT[LANG]['H_RISKS']}\n- —\n\n"
                f"### {TXT[LANG]['H_QUESTIONS']}\n- —\n\n"
                f"### {TXT[LANG]['H_NEXT']}\n- —\n"
            )
        else:
            status.update(label=TXT[LANG]["status_format"])

        rendered = canonicalize_minutes(raw_out, LANG)
        if polish:
            rendered = re.sub(r'\n{3,}', '\n\n', rendered).strip() + "\n"

        status.update(label=TXT[LANG]["status_done"], state="complete")

    st.subheader(TXT[LANG]["minutes_heading"])
    st.markdown(rendered)

    with st.expander(TXT[LANG]["mask_summary"], expanded=False):
        st.write(("- Emails masked: " if LANG=="en" else "- E-Mails anonymisiert: ") + str(email_cnt))
        st.write(("- Phones masked: " if LANG=="en" else "- Telefonnummern anonymisiert: ") + str(phone_cnt))
        if name_map:
            names_list = [name_map[k] for k in name_map]
            label = "- Names masked: " if LANG=="en" else "- Namen anonymisiert: "
            st.write(f"{label}{len(name_map)} → {names_list}")
        else:
            st.write("- Names masked: 0" if LANG=="en" else "- Namen anonymisiert: 0")
