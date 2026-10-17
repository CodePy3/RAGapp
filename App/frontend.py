import streamlit as st # used for GUI
import requests # makes calls to backend

backend_url =  "http://localhost:8000" # URL that accesses the backend

st.set_page_config(
    page_title="Chat with your PDF",
    page_icon="🤖",
    layout="wide"
) # creates page settings

st.title("Chat with your PDF's 📄")
st.caption("Powered by RAG - Retrieval Augmented Generation")

if "uploader_key" not in st.session_state: # only set on first run, so the value survives st.rerun()
    st.session_state.uploader_key = 0 # changing this key resets the file chip after DB is cleared

with st.sidebar: # creates side bar
    st.header("Your Documents")

    uploaded_file =  st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload any PDF and start asking questions",
        key=f"uploader_{st.session_state.uploader_key}",
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
        status = requests.get(f"{backend_url}/healthCheck").json()
        st.metric("chunks in database", status["total_chunks"])
    except:
        st.error("Backend is not reachable")

    st.divider()

    if "messages" not in st.session_state: # these 2 statements create a small history of chat so it doesn't refresh everytime
        st.session_state.messages = []

    for msg in st.session_state.messages: # use for loop to render previous messages
        with st.chat_message(msg["role"]):
            st.write(msg["content"]) # shows content of messages

        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("View retrieved chunks", expanded=False):
                for source in msg["sources"]:
                    st.markdown(
                        f"📎 **{source['source']}** - Page {source['page']} "
                        f"*(similarity score: {source['score']})*"
                    )

    st.divider()

    if st.button("Clear PDF's From Database ⚠️", type="secondary", use_container_width=True): # if clear button is pressed
        with st.spinner("Clearing Database..."): # runs spinner until 
            response = requests.post( # contacts backend 
                                f"{backend_url}/clearDB"
                                )
            if response.status_code == 200:
                st.session_state.uploader_key += 1
                st.rerun(scope="app")

            else:
                st.info("Failed To Clear PDF's From Database ❌")


    if st.button("Clear Chat History", type="secondary", use_container_width=True): # if clear button is pressed
            with st.spinner("Clearing Chat History..."): # runs spinner until 

                st.session_state.messages = []
            st.rerun(scope="app")


if question := st.chat_input("Ask anything about your documents"): # := (walrus operator) operator assigns chat input to variable and checks its not empty
    st.session_state.messages.append({"role":"user","content":question})
    with st.chat_message("user"):
        st.write(question)
                
    with st.chat_message("assistant"):
        with st.spinner("Searching your document..."):
            try:
                response = requests.post(
                    f"{backend_url}/ask",
                    json={"question":question,"n_results":3}  
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