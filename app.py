import time
import json
import streamlit as st
from openai import OpenAI
from composio import Composio

st.set_page_config(page_title="Composio Gmail via OpenAI", page_icon="📧", layout="centered")
st.title("📧 Send Email via Composio + OpenAI (LLM Tools)")

# --- Secrets / Clients ---
def get_clients():
    try:
        openai_key = st.secrets["OPENAI_API_KEY"]
        composio_key = st.secrets["COMPOSIO_API_KEY"]
    except KeyError as e:
        st.error(f"Missing secret: {e}. Add it to .streamlit/secrets.toml.")
        st.stop()
    return OpenAI(base_url="https://api.aimlapi.com/v1",api_key=openai_key), Composio(api_key=composio_key)

client, composio = get_clients()

# --- Sidebar: Quick help ---
with st.sidebar:
    st.header("Setup")
    st.markdown("- Put keys in **.streamlit/secrets.toml**")
    st.markdown("- Install **requirements.txt**")
    st.markdown("- Click **Start OAuth** → authorize → come back → click **I finished OAuth**")

# --- User inputs ---
st.subheader("Step 1 · Connect your Gmail account via Composio OAuth")
user_email = st.text_input(
    "Your user ID (email used as Composio user_id)",
    value="thanthtoosan.mechatronic@gmail.com",
    placeholder="you@example.com",
)
auth_config_id = st.text_input(
    "Composio auth_config_id",
    value="ac_LWCqYV-0VfOi",
    help="This must match the OAuth configuration you set in Composio.",
)

if "connection_request" not in st.session_state:
    st.session_state.connection_request = None
if "connected_account" not in st.session_state:
    st.session_state.connected_account = None
if "redirect_url" not in st.session_state:
    st.session_state.redirect_url = None

col_a, col_b = st.columns(2)
with col_a:
    start = st.button("🔗 Start OAuth")

with col_b:
    finished = st.button("✅ I finished OAuth")

if start:
    try:
        req = composio.connected_accounts.initiate(
            user_id=user_email,
            auth_config_id=auth_config_id,
        )
        st.session_state.connection_request = req
        st.session_state.redirect_url = req.redirect_url
        st.success("OAuth flow started. Use the link below to authorize.")
    except Exception as e:
        st.error(f"Failed to start OAuth: {e}")

if st.session_state.redirect_url:
    st.markdown(f"**Authorize here:** [Open OAuth link]({st.session_state.redirect_url})")
    st.info("After authorizing, return here and click **I finished OAuth**.")

if finished:
    if not st.session_state.connection_request:
        st.warning("You need to start OAuth first.")
    else:
        with st.spinner("Waiting for Composio to confirm the connection..."):
            try:
                # Blocks until connection is established (or raises on timeout/err).
                connected = st.session_state.connection_request.wait_for_connection()
                st.session_state.connected_account = connected
                st.success("Connected! Gmail tools are now available.")
            except Exception as e:
                st.error(f"Failed to confirm connection: {e}")

# --- Step 2: Compose Email ---
st.subheader("Step 2 · Compose & Send Email via LLM Tool Call")
col1, col2 = st.columns(2)
with col1:
    to_email = st.text_input("To", value="timmythaw17@gmail.com")
with col2:
    subject = st.text_input("Subject", value="Hello from composio 👋🏻")

body = st.text_area(
    "Body",
    value="Congratulations on sending your first email using AI Agents and Composio!",
    height=140,
)

confirm_send = st.checkbox("I confirm I want to send this email.")
send_btn = st.button("📤 Send Email")

def render_json(title, data):
    with st.expander(title, expanded=False):
        st.code(json.dumps(data, indent=2), language="json")

if send_btn:
    if not st.session_state.connected_account:
        st.error("Please finish OAuth connection first.")
        st.stop()
    if not confirm_send:
        st.warning("Please tick the confirmation checkbox before sending.")
        st.stop()

    # 1) Fetch the pre-configured Gmail tool from Composio for this user
    try:
        tools = composio.tools.get(
            user_id=user_email,
            tools=["GMAIL_SEND_EMAIL"],  # pre-configured tool id
        )
        if not tools:
            st.error("No tools returned. Ensure the Gmail tool is configured in Composio.")
            st.stop()
        render_json("Composio Tools", tools)
    except Exception as e:
        st.error(f"Failed to get Composio tools: {e}")
        st.stop()

    # 2) Ask OpenAI to use the tool by providing it in the tool schema
    system_msg = {"role": "system", "content": "You are a helpful assistant."}
    user_msg = {
        "role": "user",
        "content": (
            f"Send an email to {to_email} with the subject '{subject}' and "
            f"the body '{body}'"
        ),
    }

    with st.spinner("Calling OpenAI to plan the tool call..."):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-4o",
                tools=tools,  # tool schema from Composio
                messages=[system_msg, user_msg],
            )
            render_json("OpenAI Response", response.model_dump() if hasattr(response, "model_dump") else response)
        except Exception as e:
            st.error(f"OpenAI call failed: {e}")
            st.stop()

    # 3) Execute the tool calls via Composio’s provider
    with st.spinner("Executing tool calls via Composio..."):
        try:
            result = composio.provider.handle_tool_calls(
                response=response,
                user_id=user_email,
            )
            st.success("Email sent successfully!")
            render_json("Execution Result", result)
        except Exception as e:
            st.error(f"Tool execution failed: {e}")

# --- Footer ---
st.divider()
st.caption("Tip: if wait feels long, verify your OAuth status in Composio dashboard, then click **I finished OAuth**.")
