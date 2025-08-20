import json
import streamlit as st
from openai import OpenAI
from composio import Composio

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
                "You are a quiz generator. Produce high-quality multiple-choice questions (MCQs) for the given topic and difficulty.\n"
                "Return STRICT JSON only:\n\n"
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
                "}\n\n"
                "- No code fences, no commentary — JSON only.\n"
                "- Choices must be plausible and mutually exclusive.\n"
                "- Exactly one correct answer per question.\n"
            )
        }
        usr = {
            "role": "user",
            "content": f"Topic: {topic}\nDifficulty: {difficulty}\nNumber of questions: {count}\n"
        }

        with st.spinner("Generating quiz with LLM..."):
            try:
                resp = client.chat.completions.create(
                    model="openai/gpt-5-chat-latest",
                    messages=[sys, usr],
                    temperature=0.7,
                )
                raw = resp.choices[0].message.content.strip()
                quiz_obj = json.loads(raw)
                ss.quiz_json = quiz_obj
                ss.quiz_text = quiz_to_text(quiz_obj)
                ss.quiz_meta = {"topic": topic, "difficulty": difficulty, "count": int(count)}
                st.success("✅ Quiz generated!")
                st.text_area("Preview (plain text)", ss.quiz_text, height=300)
                render_json("Quiz JSON", quiz_obj)
            except json.JSONDecodeError:
                st.error("The model did not return valid JSON. Try again.")
                render_json("Raw model output", {"content": resp.choices[0].message.content})
            except Exception as e:
                st.error(f"OpenAI error: {e}")

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
    system_msg = {"role": "system", "content": "You are a helpful assistant that uses provided tools."}
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
            result = composio.provider.handle_tool_calls(response=resp, user_id=user_id)
            st.success("✅ Email sent!")
            render_json("Execution Result", result)
        except Exception as e:
            st.error(f"Tool execution failed: {e}")
