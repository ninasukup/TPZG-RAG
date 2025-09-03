import streamlit as st
import requests

# FastAPI endpoint
API_URL = "http://localhost:8000/v1/chat/completions"  # change if different

st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("🔎 TPZG RAG Assistant")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_query = st.text_input("Enter your question:")

if st.button("Ask") and user_query.strip():
    with st.spinner("Fetching answer..."):
        try:
            # Call FastAPI RAG service
            payload = {"prompt": user_query, "model": "local-llama3-8b-reranked"}
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                data = response.json()
                llm_response = data["choices"][0]["message"]["content"]

                # Save to chat history
                st.session_state.chat_history.append(
                    {"query": user_query, "response": llm_response}
                )
            else:
                st.error(f"API Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"Error: {e}")

# Display chat history
if st.session_state.chat_history:
    st.subheader("Chat History")
    for chat in st.session_state.chat_history[::-1]:
        st.markdown(f"**You:** {chat['query']}")
        st.markdown(f"**Assistant:** {chat['response']}")
        st.markdown("---")