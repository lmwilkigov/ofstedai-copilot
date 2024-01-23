import streamlit as st
from utils import init_session_state

init_session_state()

st.title("Ofsted AI Copilot")

st.markdown(
    """Welcome to the Ofsted AI Copilot. This app is designed to help you find and analyse Ofsted reports for schools.
            
To get started search for a collection of Schools on the [Search](/Search) page. 
This will load reports for an AI Chat interface on the [Chat](/Chat) page.
"""
)
