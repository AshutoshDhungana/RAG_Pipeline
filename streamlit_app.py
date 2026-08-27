import streamlit as st
import requests

API = "http://127.0.0.1:8000"
st.set_page_config(page_title="RAG Chatbot", page_icon="📄")
st.title("RAG Chatbot")
st.caption("Upload a PDF then chat.")

with st.sidebar:
    st.header("Upload PDF")
    f = st.file_uploader("Choose a PDF", type=["pdf"])
    if f and st.button("Upload"):
        try:
            r = requests.post(f"{API}/upload", files={"file": (f.name, f.getvalue(), "application/pdf")}, timeout=120)
            if r.status_code == 200:
                st.json(r.json())
            else:
                st.error(f"Upload failed (HTTP {r.status_code})")
                st.code(r.text[:2000])
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API at " + API + ". Is `uvicorn app:app` running?")
        except Exception as e:
            st.error(f"Upload error: {e}")
            st.code(r.text[:2000] if 'r' in dir() else "")

q = st.chat_input("Ask about the document...")
if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

if q:
    st.chat_message("user").write(q)
    try:
        r = requests.post(f"{API}/ask", json={"question": q}, timeout=120)
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if "error" in data:
            ans = f"API error: {data['error']}"
        else:
            ans = data.get("answer", r.text)
    except requests.exceptions.ConnectionError:
        ans = "Cannot reach the API at " + API + ". Is `uvicorn app:app` running?"
    except Exception as e:
        ans = f"Error: {e}"
    st.chat_message("assistant").write(ans)
    st.session_state.history += [("user", q), ("assistant", ans)]
