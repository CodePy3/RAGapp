import streamlit as st # used for GUI
import requests # makes calls to backend

backend_url =  "https://localhost8000"

st.set_page_config(
    page_title="Chat with your PDF",
    page_icon="🤖",
    layout="wide"
) # creates page settings

st.title("Chat with your PDF's 📄")
st.caption("Powered by RAG - Retrieval Augmented Generation")

with st.sidebar: # creates side bar
    st.header("Your Documents")

    uploaded_file =  st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload any PDF and start asking questions"
    ) # creates button used for uploading PDF 

    if uploaded_file: # if uploaded file exists 
        if st.button("Ingest PDF", type="primary", use_container_width=True): # if button is pressed
            with st.spinner("Reading, chunking and embedding your pdf"): # runs spinner until 
                response = requests.post( # contacts backend 
                    f"{backend_url}/ingest", 
                    files={"file":(uploaded_file.name,uploaded_file,"application/pdf")} 
                )

            if response.status_code == 200: # code 200 indicates successful request
                data = response.json()
                st.success(f"Done! Added {data['chunks_added']} chunks")
                st.info(f"Total in database: {data['total_chunks']} chunks")
 
            else:
                st.error("Something went wrong. Is the backend running?")

    st.divider()

    try:
        status = requests.get(f"{backend_url}/").json()
        st.metric("chunks in database", status["total chunks"])
    except:
        st.error("Backend is not reachable")

    st.divider()

    if "messages" not in st.session_state: # these 2 statements create a small history of chat so it doesn't refresh everytime
        st.session_state.messages = []

    for msg in st.session_state: # use for loop to render previous messages
        with st.chat_message(msg["role"]):
            st.write(msg["content"]) # shows content of messages

        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("View retrieved chunks", expanded=False):
                for source in msg["sources"]:
                    st.markdown(
                        f"📎 **{source['source']}** - Page {source['page']} "
                        f"*(similarity score: {source['score']})*"
                    )

    if question := st.chat_input("Ask anything about your documents"): # := (walrus operator) operator assigns chat input to variable and checks its not empty
        st.session_state.messages.append({"role":"user","content":question})
        with st.chat_message("user"):
            st.write(question)
                    
        with st.chat_message("assistant"):
            with st.spinner("Searching your document..."):
                try:
                    response = requests.post(
                        f"{backend_url}/ask",
                        json={"question":question,n_results}  
                    )

                    data = response.json()

                    if "error" in data:
                        answer = f"{data['error']}"
                        sources = []
                    else:
                        answer = data["answer"]
                        sources = data["sources"]

                except Exception as e:
                    answer = f"Could not reach the backend {e}"
                    sources = [] 

                st.write(answer)

st.session_state.messages.append({
    "role": "assistant",
    "content": answer,
    "sources": sources,
})