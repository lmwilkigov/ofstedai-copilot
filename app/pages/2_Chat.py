from datetime import datetime

import streamlit as st
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from prompts import CONDENSE_QUESTION_PROMPT, STUFF_DOCUMENT_PROMPT, WITH_SOURCES_PROMPT
from utils import (
    StreamlitStreamHandler,
    avatar_map,
    create_initial_chat_prompt,
    init_session_state,
    refresh_files,
    render_citation_response,
    show_chat_history,
)

from ofstedai.models.chat import ChatMessage

init_session_state()

st.title("Ofsted AI Copilot - Chat 🦉")

CORE_OFSTED_PROMPT = """You are Ofsted Copilot"""


st.session_state["files"] = []
st.session_state["file_uuid_map"] = {}
st.session_state["file_uuid_to_name_map"] = {}
st.session_state["school_name_to_file_uuid_map"] = {}


refresh_files()

doc_retrieval_k = 10

clear_chat = st.sidebar.button("Clear Chat")


on = st.sidebar.toggle("Toggle to focus on one school")

if on:
    school_select = st.sidebar.selectbox(
        label="Select School to chat over:",
        options=list(st.session_state.school_name_to_file_uuid_map.keys()),
    )

    parent_file_uuid_list = st.session_state.school_name_to_file_uuid_map[
        school_select[0]
    ]
else:
    # st.write("OFF. Chatting over all documents.")
    parent_file_uuid_list = []


INITIAL_CHAT_PROMPT = create_initial_chat_prompt(
    CORE_OFSTED_PROMPT,
    initial_message="Hi, I'm Ofsted Copilot. I can answer questions based on the Schools you've searched. What would you like to know?",
)


if "messages" not in st.session_state or clear_chat:
    st.session_state["messages"] = INITIAL_CHAT_PROMPT
    # clear feedback
    for key in list(st.session_state.keys()):
        if key.startswith("feedback_"):
            del st.session_state[key]

if "ai_message_markdown_lookup" not in st.session_state:
    st.session_state["ai_message_markdown_lookup"] = {}

now_formatted = datetime.now().isoformat().replace(".", "_")


show_chat_history()


def answer_question(
    question, chat_history, parent_file_uuid_list=[], callbacks=[], k=5
):
    docs_with_sources_chain = load_qa_with_sources_chain(
        st.session_state.llm,
        chain_type="stuff",
        prompt=WITH_SOURCES_PROMPT,
        document_prompt=STUFF_DOCUMENT_PROMPT,
        verbose=True,
    )

    condense_question_chain = LLMChain(
        llm=st.session_state.llm, prompt=CONDENSE_QUESTION_PROMPT
    )

    standalone_question = condense_question_chain(
        {
            "question": question,
            "chat_history": chat_history,
        }
    )["text"]

    search_kwargs = {
        "k": k,
    }

    if len(parent_file_uuid_list) > 0:
        search_kwargs["filter"] = {"parent_file_uuid": x for x in parent_file_uuid_list}

    docs = st.session_state.vector_store.as_retriever(
        search_kwargs=search_kwargs,
    ).get_relevant_documents(
        standalone_question,
    )

    result = docs_with_sources_chain(
        {
            "question": standalone_question,
            "input_documents": docs,
        },
        callbacks=callbacks,
    )

    return (result, docs_with_sources_chain)


if prompt := st.chat_input():
    st.session_state.messages.append(
        ChatMessage(
            chain=None,
            message=HumanMessage(content=prompt),
            creator_user_uuid="dev",
        )
    )
    st.chat_message("user", avatar=avatar_map["user"]).write(prompt)

    with st.chat_message("assistant", avatar=avatar_map["assistant"]):
        response_stream_text = st.empty()

        response, chain = answer_question(
            question=prompt,
            chat_history=st.session_state.messages,
            parent_file_uuid_list=parent_file_uuid_list,
            k=doc_retrieval_k,
            callbacks=[
                StreamlitStreamHandler(
                    text_element=response_stream_text, initial_text=""
                ),
            ],
        )

        response_final_markdown = render_citation_response(response)

        response_stream_text.empty()
        response_stream_text.markdown(response_final_markdown, unsafe_allow_html=True)

    st.session_state.messages.append(
        ChatMessage(
            chain=chain,
            message=AIMessage(content=response["output_text"]),
            creator_user_uuid="dev",
        )
    )

    # Store the markdown response for later rendering
    st.session_state.ai_message_markdown_lookup[
        hash(response["output_text"])
    ] = response_final_markdown
