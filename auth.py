import streamlit as st
import os

def get_api_key(key_name: str = "API_KEY") -> str | None:
    """
    Retrieves API key from (in order):
    1. Streamlit Secrets (Cloud)
    2. Environment Variables (Local/CI)
    3. Streamlit Session State (User input)
    """

    # 1. Streamlit Cloud secrets
    key = st.secrets.get(key_name)

    # 2. Environment variable fallback
    if not key:
        key = os.getenv(key_name)

    # 3. Session state (user entered)
    if not key:
        key = st.session_state.get(key_name)

    return key


def request_api_key_ui(key_name: str = "API_KEY"):
    """
    Prompts the user to enter the API key securely in the UI.
    """
    st.info("Please enter your API key to continue.")
    entered = st.text_input(
        "API Key",
        type="password",
        placeholder="Paste your API key here"
    )

    if entered:
        st.session_state[key_name] = entered
        st.success("API key saved for this session.")
        return entered

    st.stop()
