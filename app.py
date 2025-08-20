import re, json
from typing import List
from pydantic import BaseModel, ValidationError

class Question(BaseModel):
    question: str
    choices: List[str]
    correctIndex: int
    explanation: str

class Quiz(BaseModel):
    title: str
    instructions: str
    questions: List[Question]

def extract_json_block(text: str) -> str:
    """Strip code fences and return the substring between the first '{' and last '}'."""
    if not text:
        raise ValueError("Empty model output.")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find JSON object in the output.")
    return cleaned[start:end+1]

import json
import tempfile
from pathlib import Path
import streamlit as st
from openai import OpenAI
from composio import Composio
from quiz_agent import (
    SMTPConfig,
    generate_quiz_from_pdf,
    agent_mode_send_quiz,
)

# ---------- Page setup ----------
st.set_page_config(page_title="Quiz + Email via Composio", page_icon="🧩", layout="wide")
st.title("🧩 Quiz Generator + 📧 Email Sender (Composio + OpenAI)")

# ---------- Secrets / Clients ----------
@st.cache_resource
def _get_clients():
    try:
        openai_key = st.secrets["OPENAI_API_KEY"]
        composio_key = st.secrets["COMPOSIO_API_KEY"]
    except KeyError as e:
        st.error(f"Missing secret: {e}. Add it to .streamlit/secrets.toml and restart.")
        st.stop()
    return OpenAI(base_url="https://api.aimlapi.com/v1",api_key=openai_key), Composio(api_key=composio_key)

client, composio = _get_clients()

# ---------- Session defaults ----------
ss = st.session_state
ss.setdefault("connection_request", None)
ss.setdefault("connected_account", None)
ss.setdefault("redirect_url", None)
ss.setdefault("quiz_json", None)
ss.setdefault("quiz_text", "")
ss.setdefault("quiz_meta", {"topic": "", "difficulty": "", "count": 0})
ss.setdefault("pdf_quiz_text", "")
ss.setdefault("last_pdf_path", None)

# ---------- Sidebar: Composio auth ----------
with st.sidebar:
    st.header("🔐 Composio Connection")

    user_id = st.text_input(
        "User ID (email used as Composio user_id)",
        value="",
        placeholder="you@example.com",
        help="This will identify your connected account in Composio."
    )
    auth_config_id = st.text_input(
        "Composio auth_config_id",
        value="ac_LWCqYV-0VfOi",
        help="Use the OAuth config ID from your Composio dashboard."
    )

    col_sb1, col_sb2 = st.columns(2)
    start_oauth = col_sb1.button("🔗 Start OAuth")
    finish_oauth = col_sb2.button("✅ I finished OAuth")

    if start_oauth:
        if not user_id or not auth_config_id:
            st.warning("Please fill both User ID and auth_config_id.")
        else:
            try:
                req = composio.connected_accounts.initiate(
                    user_id=user_id,
                    auth_config_id=auth_config_id,
                )
                ss.connection_request = req
                ss.redirect_url = req.redirect_url
                st.success("OAuth started. Click the link below to authorize.")
            except Exception as e:
                st.error(f"Failed to start OAuth: {e}")

    if ss.redirect_url:
        st.markdown(f"[👉 Open OAuth link to authorize Gmail access]({ss.redirect_url})")
        st.info("After authorizing, return here and click **I finished OAuth**.")

    if finish_oauth:
        if not ss.connection_request:
            st.warning("Start OAuth first.")
        else:
            with st.spinner("Waiting for Composio to confirm the connection..."):
                try:
                    connected = ss.connection_request.wait_for_connection()
                    ss.connected_account = connected
                    st.success("✅ Connected! Gmail tool is available.")
                except Exception as e:
                    st.error(f"Failed to confirm connection: {e}")

    st.divider()
    st.caption("Tips:\n- Keep this window open during OAuth\n- Make sure the Gmail tool is configured in Composio")

# ---------- Helpers ----------
def render_json(label, data):
    with st.expander(label, expanded=False):
        try:
            st.code(json.dumps(data, indent=2), language="json")
        except TypeError:
            st.write(data)

def quiz_to_text(quiz_obj: dict) -> str:
    lines = []
    title = quiz_obj.get("title", "Quiz")
    instructions = quiz_obj.get("instructions", "")
    lines.append(title)
    if instructions:
        lines.append(instructions)
    lines.append("")

    for i, q in enumerate(quiz_obj.get("questions", []), start=1):
        lines.append(f"{i}. {q.get('question','')}")
        choices = q.get("choices", [])
        for idx, ch in enumerate(choices, start=1):
            lines.append(f"   {chr(64+idx)}. {ch}")  # A., B., C., ...
        # Show answer as well:
        ci = q.get("correctIndex", None)
        if isinstance(ci, int) and 0 <= ci < len(choices):
            lines.append(f"   ✅ Answer: {chr(65+ci)}")
        expl = q.get("explanation", "")
        if expl:
            lines.append(f"   ℹ️  {expl}")
        lines.append("")
    return "\n".join(lines)

# ---------- Main: 1) Generate quiz ----------
st.header("1) Generate a Quiz with LLM")
col1, col2, col3 = st.columns([2,1,1])
with col1:
    topic = st.text_input("Topic", value=ss.quiz_meta.get("topic", ""))
with col2:
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
with col3:
    count = st.number_input("Number of Questions", min_value=1, max_value=50, value=5, step=1)

gen_btn = st.button("🧠 Generate Quiz")

if gen_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
    else:
        sys = {
            "role": "system",
            "content": (
                "You are a quiz generator. Produce high-quality MCQs.\n"
                "Return STRICT JSON only with this exact shape:\n"
                "{\n"
                '  "title": string,\n'
                '  "instructions": string,\n'
                '  "questions": [\n'
                "    {\n"
                '      "question": string,\n'
                '      "choices": [string, string, string, string],\n'
                '      "correctIndex": number,\n'
                '      "explanation": string\n'
                "    }\n"
                "  ]\n"
                "}\n"
                "- No code fences. No commentary. JSON object only."
            )
        }
        usr = {
            "role": "user",
            "content": f"Topic: {topic}\nDifficulty: {difficulty}\nNumber of questions: {count}\n"
        }

        with st.spinner("Generating quiz with LLM..."):
            last_err = None
            quiz_obj = None
            # Try up to 2 times: first with strict JSON enforcement, then fallback
            for attempt in range(2):
                try:
                    resp = client.chat.completions.create(
                        model="openai/gpt-5-chat-latest",
                        messages=[sys, usr],
                        temperature=0,
                        response_format={"type": "json_object"},  # hard-enforce JSON
                    )
                except Exception as e:
                    st.error(f"OpenAI API error: {e}")
                    break

                raw = resp.choices[0].message.content or ""

                # Parse -> validate
                try:
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        data = json.loads(extract_json_block(raw))

                    # Validate shape
                    quiz_valid = Quiz.model_validate(data)
                    quiz_obj = quiz_valid.model_dump()
                    break  # success
                except (json.JSONDecodeError, ValidationError, ValueError) as e:
                    last_err = e
                    # On second failure, give up
                    if attempt == 1:
                        pass

            if not quiz_obj:
                st.error("The model did not return valid JSON. I tried to coerce it but failed.")
                st.caption(f"Parser note: {last_err}")
                st.text_area("Raw model output", raw, height=240)
            else:
                ss.quiz_json = quiz_obj
                # your existing quiz_to_text() is fine:
                ss.quiz_text = quiz_to_text(quiz_obj)
                ss.quiz_meta = {"topic": topic, "difficulty": difficulty, "count": int(count)}
                st.success("✅ Quiz generated!")
                st.text_area("Preview (plain text)", ss.quiz_text, height=300)
                with st.expander("Quiz JSON"):
                    st.code(json.dumps(quiz_obj, indent=2), language="json")

# ---------- Main: 2) Email the quiz via Composio Gmail tool ----------
st.header("2) Email the Generated Quiz")

to_email = st.text_input("Recipient email", value="", placeholder="recipient@example.com")
default_subject = f"Quiz: {ss.quiz_meta.get('topic','(no topic)')} — {ss.quiz_meta.get('difficulty','')}"
subject = st.text_input("Subject", value=default_subject)
body_prefill = ss.quiz_text or "Generate a quiz first, then come back here."
body = st.text_area("Email body", value=body_prefill, height=260)
confirm_send = st.checkbox("I confirm I want to send this email.")
send_btn = st.button("📤 Send via Composio + LLM Tool Call")

if send_btn:
    if not ss.connected_account:
        st.error("Please complete Composio OAuth in the sidebar first.")
        st.stop()
    if not to_email.strip():
        st.warning("Please enter a recipient email.")
        st.stop()
    if not confirm_send:
        st.warning("Please tick the confirmation checkbox.")
        st.stop()

    # Get Gmail tool schema from Composio
    try:
        tools = composio.tools.get(user_id=user_id, tools=["GMAIL_SEND_EMAIL"])
        if not tools:
            st.error("No Gmail tool found. Ensure it's configured in Composio.")
            st.stop()
        render_json("Composio Tools", tools)
    except Exception as e:
        st.error(f"Failed to fetch tools: {e}")
        st.stop()

    # Ask the LLM to use the tool to send email
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant that MUST use the provided tools. "
            "When asked to send an email, you MUST call the Gmail tool with the correct fields. "
            "Do not respond with natural language; only produce the required tool calls."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Send an email to {to_email} with the subject '{subject}' and the body below.\n\n{body}"
        ),
    }

    with st.spinner("Planning tool call with OpenAI..."):
        try:
            resp = client.chat.completions.create(
                model="openai/gpt-5-chat-latest",
                tools=tools,
                tool_choice="auto",
                messages=[system_msg, user_msg],
                temperature=0,
            )
            render_json("OpenAI Response", resp.model_dump() if hasattr(resp, "model_dump") else {"response": str(resp)})
        except Exception as e:
            st.error(f"OpenAI call failed: {e}")
            st.stop()

    # Execute tool calls via Composio
    with st.spinner("Executing tool call via Composio..."):
        try:
            # If the model didn't return any tool calls, inform the user early
            try:
                tool_calls = resp.choices[0].message.tool_calls  # type: ignore[attr-defined]
            except Exception:
                tool_calls = None

            if not tool_calls:
                st.warning("Model returned no tool calls. Ensure OAuth is complete and try again.")
                # Show model message for debugging
                render_json("Model Message", resp.choices[0].message.model_dump() if hasattr(resp.choices[0].message, "model_dump") else {"message": str(resp.choices[0].message)})
            else:
                result = composio.provider.handle_tool_calls(response=resp, user_id=user_id)
                st.success("✅ Email sent!")
                render_json("Execution Result", result)
        except Exception as e:
            st.error(f"Tool execution failed: {e}")

st.divider()

# ---------- Main: 3) Generate Quiz from PDF and Send via SMTP ----------
st.header("3) Generate Quiz from PDF and Send via SMTP")

col_pdf1, col_pdf2 = st.columns([2, 1])
with col_pdf1:
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
with col_pdf2:
    pdf_num_q = st.number_input("Questions (PDF)", min_value=1, max_value=30, value=5, step=1)
    pdf_seed = st.number_input("Seed (optional)", min_value=0, value=0, step=1)
    seed_value = int(pdf_seed) if pdf_seed else None

pdf_gen_btn = st.button("📄 Generate Quiz from PDF")

if pdf_gen_btn:
    if not uploaded_pdf:
        st.warning("Please upload a PDF first.")
    else:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.getbuffer())
                tmp_path = tmp.name
            ss.last_pdf_path = tmp_path

            with st.spinner("Extracting text and generating quiz..."):
                pdf_quiz = generate_quiz_from_pdf(
                    pdf_path=tmp_path,
                    num_questions=int(pdf_num_q),
                    seed=seed_value,
                )
            ss.pdf_quiz_text = pdf_quiz
            st.success("✅ PDF quiz generated!")
            st.text_area("PDF Quiz (plain text)", ss.pdf_quiz_text, height=300)
        except Exception as e:
            st.error(f"Failed to generate quiz from PDF: {e}")

with st.expander("SMTP Settings", expanded=False):
    col_smtp1, col_smtp2 = st.columns(2)
    with col_smtp1:
        smtp_host = st.text_input("SMTP Host", value="")
        smtp_port = st.number_input("SMTP Port", min_value=1, max_value=65535, value=587, step=1)
        smtp_use_tls = st.checkbox("Use STARTTLS (disable to use SSL)", value=True)
    with col_smtp2:
        smtp_username = st.text_input("SMTP Username", value="")
        smtp_password = st.text_input("SMTP Password", value="", type="password")

col_mail1, col_mail2 = st.columns([2, 1])
with col_mail1:
    to_email_pdf = st.text_input("Recipient email (SMTP)", value="", placeholder="recipient@example.com")
with col_mail2:
    subject_pdf = st.text_input("Subject (SMTP)", value="Quiz from your PDF")

send_pdf_btn = st.button("📧 Send PDF Quiz via SMTP")

if send_pdf_btn:
    if not ss.last_pdf_path or not Path(ss.last_pdf_path).exists():
        st.warning("Please upload a PDF and generate a quiz first.")
    elif not to_email_pdf.strip() or not subject_pdf.strip():
        st.warning("Please enter recipient email and subject.")
    elif not smtp_host or not smtp_port or not smtp_username or not smtp_password:
        st.warning("Please fill all SMTP settings.")
    else:
        try:
            cfg = SMTPConfig(
                host=smtp_host,
                port=int(smtp_port),
                username=smtp_username,
                password=smtp_password,
                use_tls=bool(smtp_use_tls),
            )
            with st.spinner("Sending email via SMTP..."):
                agent_mode_send_quiz(
                    pdf_path=ss.last_pdf_path,
                    to_email=to_email_pdf,
                    subject=subject_pdf,
                    smtp_config=cfg,
                    num_questions=int(pdf_num_q),
                    seed=seed_value,
                )
            st.success("✅ Email sent via SMTP!")
        except Exception as e:
            st.error(f"Failed to send email: {e}")
