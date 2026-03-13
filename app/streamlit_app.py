import json
import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

ROOT = Path(__file__).resolve().parents[1]
VECTOR_INFO = ROOT / "ingestion" / "vector_store_info.json"
METADATA_CSV = ROOT / "metadata" / "guidelines.csv"

# Update this if your repository name changes.
GITHUB_REPO_OWNER = "mcgovernandy"
GITHUB_REPO_NAME = "clinical-guideline-assistant"
GITHUB_BRANCH = "main"
RAW_GITHUB_BASE = f"https://raw.githubusercontent.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/{GITHUB_BRANCH}/guidelines"

PREFERRED_LEVELS = ["local", "national", "international"]
LEVEL_LABELS = {
    "local": "Local guidance",
    "national": "National guidance",
    "international": "International guidance",
}

LEVEL_RESPONSE_SCHEMA = {
    "name": "level_guideline_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "supported": {"type": "boolean"},
            "level": {"type": "string"},
            "summary": {"type": "string"},
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "section_heading": {"type": "string"},
                        "answer": {"type": "string"},
                        "follow_up_needed": {"type": "boolean"},
                        "notes": {"type": "string"},
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "quote": {"type": "string"},
                                    "context_before": {"type": "string"},
                                    "context_after": {"type": "string"}
                                },
                                "required": ["quote", "context_before", "context_after"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": [
                        "source",
                        "section_heading",
                        "answer",
                        "follow_up_needed",
                        "notes",
                        "evidence"
                    ],
                    "additionalProperties": False
                }
            },
            "overall_notes": {"type": "string"}
        },
        "required": ["supported", "level", "summary", "recommendations", "overall_notes"],
        "additionalProperties": False
    }
}

SYNTHESIS_SCHEMA = {
    "name": "hierarchical_guideline_synthesis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "supported": {"type": "boolean"},
            "headline": {"type": "string"},
            "main_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "evidence_keys": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["text", "evidence_keys"],
                    "additionalProperties": False
                }
            },
            "differences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_level": {"type": "string"},
                        "text": {"type": "string"},
                        "evidence_keys": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["source_level", "text", "evidence_keys"],
                    "additionalProperties": False
                }
            },
            "clinical_caveat": {"type": "string"},
            "disclaimer": {"type": "string"}
        },
        "required": ["supported", "headline", "main_points", "differences", "clinical_caveat", "disclaimer"],
        "additionalProperties": False
    }
}

with open(VECTOR_INFO, encoding="utf-8") as f:
    vector_info = json.load(f)

metadata_df = pd.read_csv(METADATA_CSV).fillna("")


def normalise_text(value: str) -> str:
    return str(value).strip().lower()


def get_vector_store_id(level: str) -> str | None:
    return vector_info.get("vector_stores", {}).get(level, {}).get("vector_store_id")


def get_guideline_record(source_value: str):
    if not source_value:
        return None

    source_norm = normalise_text(source_value)

    for field in ["filename", "title"]:
        exact = metadata_df[metadata_df[field].astype(str).str.strip().str.lower() == source_norm]
        if not exact.empty:
            return exact.iloc[0]

    for field in ["filename", "title"]:
        partial = metadata_df[metadata_df[field].astype(str).str.strip().str.lower().str.contains(source_norm, regex=False)]
        if not partial.empty:
            return partial.iloc[0]

    return None


def make_raw_github_url(source_value: str) -> str | None:
    record = get_guideline_record(source_value)
    if record is None:
        return None
    filename = str(record["filename"]).strip()
    return f"{RAW_GITHUB_BASE}/{quote(filename)}"


def make_guideline_label(source_value: str) -> str:
    record = get_guideline_record(source_value)
    if record is None:
        return source_value
    title = str(record.get("title", "")).strip() or str(record.get("filename", "")).strip()
    approval_date = str(record.get("approval_date", "")).strip()
    level = str(record.get("guideline_level", "")).strip().title()
    suffix = []
    if level:
        suffix.append(level)
    if approval_date:
        suffix.append(approval_date)
    if suffix:
        return f"{title} ({' | '.join(suffix)})"
    return title


def ask_level(question: str, level: str, vector_store_id: str) -> dict:
    instructions = f"""
You are Sally, a diabetes and endocrinology clinical guideline assistant.

You are currently reviewing ONLY {level.upper()} guidelines.
Use only retrieved text from file search. Do not use background medical knowledge.
Return up to 3 distinct relevant guideline recommendations from this level if they genuinely address the question.
If nothing is clearly supported, set supported=false and return an empty recommendations array.

Rules:
1. Keep answers concise, operational, and suitable for a diabetes specialist.
2. source must be the filename or recognisable guideline title.
3. answer must reflect only the wording in the retrieved guideline.
4. evidence.quote must be verbatim text from the guideline.
5. context_before and context_after should be short adjacent fragments for display context, not paraphrases.
6. If the retrieved content is incomplete for safe operational use, set follow_up_needed=true and explain why in notes.
7. Do not invent conflicts with other levels because you cannot see them in this step.
8. If this level has multiple relevant guidelines with different operational advice, include them as separate recommendations.
"""

    response = client.responses.create(
        model="gpt-5.4",
        instructions=instructions,
        input=question,
        tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}],
        text={
            "format": {
                "type": "json_schema",
                "name": LEVEL_RESPONSE_SCHEMA["name"],
                "strict": LEVEL_RESPONSE_SCHEMA["strict"],
                "schema": LEVEL_RESPONSE_SCHEMA["schema"],
            }
        },
    )
    data = json.loads(response.output_text)
    data["level"] = level
    return data


def build_evidence_index(level_results: dict) -> tuple[list[dict], dict]:
    evidence_items = []
    by_key = {}
    counter = 1

    for level in PREFERRED_LEVELS:
        result = level_results.get(level, {})
        for rec_idx, rec in enumerate(result.get("recommendations", []), start=1):
            for ev_idx, ev in enumerate(rec.get("evidence", []), start=1):
                key = f"E{counter}"
                item = {
                    "key": key,
                    "number": counter,
                    "level": level,
                    "recommendation_index": rec_idx,
                    "evidence_index": ev_idx,
                    "source": rec.get("source", ""),
                    "section_heading": rec.get("section_heading", "Not clearly identified"),
                    "quote": ev.get("quote", ""),
                    "context_before": ev.get("context_before", ""),
                    "context_after": ev.get("context_after", ""),
                }
                evidence_items.append(item)
                by_key[key] = item
                counter += 1

    return evidence_items, by_key


def synthesise_answer(question: str, level_results: dict, evidence_items: list[dict]) -> dict:
    synthesis_input = {
        "question": question,
        "preference_order": PREFERRED_LEVELS,
        "level_results": level_results,
        "evidence_catalogue": [
            {
                "key": item["key"],
                "level": item["level"],
                "source": item["source"],
                "section_heading": item["section_heading"],
                "quote": item["quote"],
            }
            for item in evidence_items
        ],
    }

    instructions = """
You are Sally (SAL 9000), a specialist diabetes clinical guideline assistant.

Your task is to synthesise already-extracted guideline findings.
Preference order is local > national > international.

Rules:
1. The main recommendation must be based on the highest available hierarchy level that supports the question.
2. If multiple local recommendations exist, preserve that nuance rather than collapsing them into one simplistic answer.
3. differences should explain where national or international guidance takes a different or broader approach.
4. Every main point and every difference must cite evidence_keys chosen ONLY from the supplied evidence catalogue.
5. Do not invent evidence keys.
6. Write in a concise advisory style suitable for a diabetes specialist.
7. disclaimer must remind the user that Sally is an aid, not a replacement for direct review of the full guideline and clinical judgement.
"""

    response = client.responses.create(
        model="gpt-5.4",
        instructions=instructions,
        input=json.dumps(synthesis_input, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": SYNTHESIS_SCHEMA["name"],
                "strict": SYNTHESIS_SCHEMA["strict"],
                "schema": SYNTHESIS_SCHEMA["schema"],
            }
        },
    )
    return json.loads(response.output_text)


def render_text_with_refs(text: str, evidence_keys: list[str], evidence_lookup: dict) -> str:
    refs = "".join(
        f" <a href='#ref-{evidence_lookup[key]['number']}' style='text-decoration:none;'>[{evidence_lookup[key]['number']}]</a>"
        for key in evidence_keys
        if key in evidence_lookup
    )
    return f"{text}{refs}"


def clean_fragment(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def render_quote_context(item: dict):
    before = clean_fragment(item.get("context_before", ""))
    quote = clean_fragment(item.get("quote", ""))
    after = clean_fragment(item.get("context_after", ""))

    html = "<div class='quote-context'>"
    if before:
        html += f"<span class='muted'>{before} </span>"
    if quote:
        html += f"<span class='quote-highlight'>“{quote}”</span>"
    if after:
        html += f" <span class='muted'>{after}</span>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_recommendation_card(rec: dict, level: str):
    source = rec.get("source", "")
    source_label = make_guideline_label(source)
    source_url = make_raw_github_url(source)

    st.markdown("<div class='recommendation-card'>", unsafe_allow_html=True)
    if source_url:
        st.markdown(f"**Source:** [{source_label}]({source_url})")
    else:
        st.markdown(f"**Source:** {source_label}")
    st.markdown(f"**Section:** {rec.get('section_heading', 'Not clearly identified')}")
    st.markdown(f"**Advice:** {rec.get('answer', '')}")
    st.markdown(f"**Needs direct review of full text:** {'Yes' if rec.get('follow_up_needed') else 'No'}")
    if rec.get("notes"):
        st.markdown(f"**Notes:** {rec['notes']}")

    evidence = rec.get("evidence", [])
    if evidence:
        st.markdown("**Quoted text**")
        for ev in evidence:
            render_quote_context(ev)
    st.markdown("</div>", unsafe_allow_html=True)


def render_level_column(level: str, result: dict):
    st.subheader(LEVEL_LABELS[level])
    if not result.get("supported"):
        st.info(f"No clearly supported answer found in the {level} guideline set.")
        return

    if result.get("summary"):
        st.markdown(f"**Level summary:** {result['summary']}")

    recommendations = result.get("recommendations", [])
    if not recommendations:
        st.info("No recommendation cards returned.")
        return

    for rec in recommendations:
        render_recommendation_card(rec, level)

    if result.get("overall_notes"):
        st.caption(result["overall_notes"])


st.set_page_config(page_title="Sally", page_icon="🔴", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        padding: 1rem 1.2rem 1rem 1.2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #081c2e 0%, #0f2742 45%, #12355b 100%);
        color: white;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #dbeafe;
    }
    .logo-wrap {
        display:flex;
        align-items:center;
        gap:0.9rem;
        margin-bottom:0.5rem;
    }
    .logo-lens {
        width:56px;
        height:56px;
        border-radius:50%;
        background: radial-gradient(circle at 50% 50%, #ff7a7a 0%, #e11d48 32%, #7f1d1d 66%, #0b1220 100%);
        box-shadow: 0 0 0 6px rgba(59,130,246,0.22), 0 0 18px rgba(239,68,68,0.65);
        border: 4px solid #60a5fa;
    }
    .recommendation-card {
        background: #ffffff;
        border: 1px solid #dbe4f0;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .quote-context {
        background: #f8fafc;
        border-left: 3px solid #60a5fa;
        padding: 0.6rem 0.8rem;
        margin: 0.35rem 0 0.6rem 0;
        border-radius: 8px;
        line-height: 1.45;
    }
    .muted {
        color: #94a3b8;
    }
    .quote-highlight {
        color: #0f172a;
        font-weight: 600;
    }
    .citation-card {
        background: #f8fafc;
        border: 1px solid #dbe4f0;
        border-radius: 12px;
        padding: 0.85rem 0.95rem;
        margin-bottom: 0.8rem;
    }
    .disclaimer {
        background: #fff7ed;
        border: 1px solid #fdba74;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='hero'>
      <div class='logo-wrap'>
        <div class='logo-lens'></div>
        <div>
          <div class='hero-title'>Sally</div>
          <div class='hero-sub'>SAL 9000 · Diabetes guideline assistant with hierarchy-aware evidence synthesis</div>
        </div>
      </div>
      <div class='hero-sub'>Local guidance is preferred. National and international recommendations are shown when they differ or add context.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Clinical safety")
    st.markdown(
        "Sally is a decision-support aid for clinicians. She can miss context, mis-rank evidence, or quote partial passages. "
        "Always review the linked source guideline, apply local policy, and use specialist clinical judgement."
    )
    st.subheader("Guideline stores")
    for level in PREFERRED_LEVELS:
        store_id = get_vector_store_id(level)
        if store_id:
            st.code(f"{level}: {store_id}")

question = st.text_area(
    "Ask Sally a clinical question",
    height=120,
    placeholder="Example: In the peri-operative setting, when should SGLT2 inhibitors be withheld and restarted?",
)

ask_button = st.button("Ask Sally")

if ask_button:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Sally is reviewing local, national, and international guidance..."):
            try:
                level_results = {}
                for level in PREFERRED_LEVELS:
                    vector_store_id = get_vector_store_id(level)
                    if vector_store_id:
                        level_results[level] = ask_level(question.strip(), level, vector_store_id)
                    else:
                        level_results[level] = {
                            "supported": False,
                            "level": level,
                            "summary": "",
                            "recommendations": [],
                            "overall_notes": "No vector store configured for this level.",
                        }

                evidence_items, evidence_lookup = build_evidence_index(level_results)
                synthesis = synthesise_answer(question.strip(), level_results, evidence_items)

                if synthesis.get("supported"):
                    st.success(synthesis.get("headline", "Supported answer found."))
                else:
                    st.warning("Sally could not find a clearly supported answer across the current guideline set.")

                st.subheader("Answer")
                for point in synthesis.get("main_points", []):
                    st.markdown(
                        render_text_with_refs(point.get("text", ""), point.get("evidence_keys", []), evidence_lookup),
                        unsafe_allow_html=True,
                    )

                if synthesis.get("differences"):
                    st.subheader("Different approaches in national/international guidance")
                    for diff in synthesis["differences"]:
                        label = LEVEL_LABELS.get(diff.get("source_level", ""), diff.get("source_level", "Other guidance").title())
                        st.markdown(
                            f"**{label}:** " + render_text_with_refs(diff.get("text", ""), diff.get("evidence_keys", []), evidence_lookup),
                            unsafe_allow_html=True,
                        )

                if synthesis.get("clinical_caveat"):
                    st.info(synthesis["clinical_caveat"])

                st.subheader("Guidance by hierarchy level")
                col1, col2, col3 = st.columns(3)
                with col1:
                    render_level_column("local", level_results.get("local", {}))
                with col2:
                    render_level_column("national", level_results.get("national", {}))
                with col3:
                    render_level_column("international", level_results.get("international", {}))

                st.subheader("References")
                if evidence_items:
                    for item in evidence_items:
                        url = make_raw_github_url(item["source"])
                        label = make_guideline_label(item["source"])
                        st.markdown(f"<div class='citation-card' id='ref-{item['number']}'>", unsafe_allow_html=True)
                        if url:
                            st.markdown(f"**[{item['number']}] [{label}]({url})**")
                        else:
                            st.markdown(f"**[{item['number']}] {label}**")
                        st.markdown(f"**Hierarchy:** {item['level'].title()}")
                        st.markdown(f"**Section:** {item['section_heading']}")
                        render_quote_context(item)
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.write("No references were returned.")

                disclaimer = synthesis.get("disclaimer") or (
                    "Sally is an AI guideline assistant and not a substitute for direct review of the original guideline, "
                    "full clinical context, or specialist judgement."
                )
                st.markdown(f"<div class='disclaimer'><strong>Disclaimer:</strong> {disclaimer}</div>", unsafe_allow_html=True)

                with st.expander("Raw structured output"):
                    st.json({"level_results": level_results, "synthesis": synthesis})

            except Exception as e:
                st.error("API call failed.")
                st.code(str(e))
