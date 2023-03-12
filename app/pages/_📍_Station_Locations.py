import streamlit as st

st.set_page_config(page_title="Station Locations", page_icon=":detective:", layout="wide")

st.markdown(
    """
    # 📍 Station Locations
    
    This section is to help you detect churners, based on a specific period of time. The output will be a list of the clients with their client id, and potentially personal information.
    """
)