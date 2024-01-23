import json
import os
import pathlib
from datetime import date
from typing import List
from collections import defaultdict

import dotenv
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models.anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, SystemMessage
from langchain.schema.output import LLMResult
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

from ofstedai.models import Chunk, File
from ofstedai.models.chat import ChatMessage
from ofstedai.storage.filesystem import FileSystemStorageHandler

# fmt: off
avatar_map = {"human": "🧑‍💻", "ai": "🦉", "user": "🧑‍💻", "assistant" : "🦉"}
# fmt: on


def init_session_state() -> dict:
    """Initialise the session state for the app"""

    # Bring VARS into environment
    dotenv.load_dotenv(".env")
    # Grab it as a dictionary too for convenience
    ENV = dotenv.dotenv_values(".env")

    if "storage_handler" not in st.session_state:
        persistency_folder_path = pathlib.Path("./data/")
        st.session_state.storage_handler = FileSystemStorageHandler(
            root_path=persistency_folder_path
        )

    if "llm" not in st.session_state:
        st.session_state.llm = ChatAnthropic(
            anthropic_api_key=ENV["ANTHROPIC_API_KEY"],
            max_tokens=500,
            temperature=0.3,
            streaming=True,
        )

    if "embedding_function" not in st.session_state:
        st.session_state.embedding_function = SentenceTransformerEmbeddings()

    if "vector_store" not in st.session_state:
        persist_directory = os.path.join("data", "VectorStore")

        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        st.session_state.vector_store = Chroma(
            embedding_function=st.session_state.embedding_function,
            persist_directory=persist_directory,
        )

    return ENV


class StreamlitStreamHandler(BaseCallbackHandler):
    """Callback handler for streamlit stream elements"""

    def __init__(self, text_element, initial_text=""):
        self.text_element = text_element
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.text_element.write(self.text)

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        self.text_element.empty()

    def sync(self):
        self.text_element.write(self.text)


def add_chunks_to_vector_store(chunks: List[Chunk]) -> None:
    """Takes a list of Chunks and embeds them into the vector store

    Args:
        chunks (List[Chunk]): The chunks to be added to the vector store
    """

    metadatas = [dict(chunk.metadata) for chunk in chunks]

    for i, chunk in enumerate(chunks):
        # add other chunk fields to metadata
        metadatas[i]["uuid"] = chunk.uuid
        metadatas[i]["parent_file_uuid"] = chunk.parent_file_uuid
        metadatas[i]["index"] = chunk.index
        metadatas[i]["created_datetime"] = chunk.created_datetime
        metadatas[i]["token_count"] = chunk.token_count
        metadatas[i]["text_hash"] = chunk.text_hash

    sanitised_metadatas = []

    for metadata in metadatas:
        for k, v in metadata.items():
            if isinstance(v, list) or isinstance(v, dict):
                # Converting {k} metadata into JSON string to make vectorstore safe
                metadata[k] = json.dumps(metadata[k], ensure_ascii=False)

        sanitised_metadatas.append(metadata)

    batch_size = 160
    for i in range(0, len(chunks), batch_size):
        st.session_state.vector_store.add_texts(
            texts=[chunk.text for chunk in chunks[i : i + batch_size]],
            metadatas=[meta for meta in sanitised_metadatas[i : i + batch_size]],
            ids=[chunk.uuid for chunk in chunks[i : i + batch_size]],
        )


def refresh_files():
    st.session_state["files"] = st.session_state.storage_handler.read_all_items(
        model_type="File"
    )
    st.session_state["file_uuid_map"] = {x.uuid: x for x in st.session_state["files"]}
    st.session_state["file_uuid_to_name_map"] = {
        x.uuid: x.name for x in st.session_state["files"]
    }

    st.session_state["school_name_to_file_uuid_map"] = defaultdict(list)

    for file in st.session_state["files"]:
        st.session_state["school_name_to_file_uuid_map"][file.school_name].append(
            file.uuid
        )




def create_initial_chat_prompt(
    CORE_PROMPT: PromptTemplate,
    initial_message: str,
) -> list[ChatMessage]:
    """Generate an INITIAL_CHAT_PROMPT."""
    INITIAL_CHAT_PROMPT = [
        ChatMessage(
            chain=None,
            message=SystemMessage(content=CORE_PROMPT),
            creator_user_uuid="dev",
        ),
        ChatMessage(
            chain=None,
            message=AIMessage(content=initial_message),
            creator_user_uuid="dev",
        ),
    ]
    return INITIAL_CHAT_PROMPT


def show_chat_history():
    for i, chat_response in enumerate(st.session_state.messages):
        msg = chat_response.message
        if msg.type == "system":
            continue

        if hash(msg.content) in st.session_state.ai_message_markdown_lookup:
            with st.chat_message(msg.type, avatar=avatar_map[msg.type]):
                st.markdown(
                    st.session_state.ai_message_markdown_lookup[hash(msg.content)],
                    unsafe_allow_html=True,
                )
        else:
            st.chat_message(msg.type, avatar=avatar_map[msg.type]).write(msg.content)


def get_files_by_uuid(file_uuids):
    files = st.session_state.storage_handler.read_items(file_uuids, "File")
    return files


def replace_doc_ref(
    output_for_render: str = "",
    files: List[File] = [],
    page_numbers: List = [],
    flexible=False,
):
    if len(page_numbers) != len(files):
        page_numbers = [None for _ in files]

    modified_text = output_for_render

    for i, file in enumerate(files):
        file_button = f"[{file.name}]({file.school_url})"

        strings_to_replace = [
            f"<Doc{file.uuid}>",
        ]
        if flexible:
            strings_to_replace += [
                f"<Doc {file.uuid}>",
                f"Doc {file.uuid}",
                f"Document {file.uuid}",
            ]

        for string_to_replace in strings_to_replace:
            modified_text = modified_text.replace(string_to_replace, file_button)
    return modified_text


def render_citation_response(response: str):
    cited_chunks = [
        (
            chunk.metadata["parent_doc_uuid"],
            chunk.metadata["filename"],
            chunk.metadata["page_numbers"]
            if "page_numbers" in chunk.metadata
            else None,
        )
        for chunk in response["input_documents"]
    ]
    cited_chunks = set(cited_chunks)
    cited_files = get_files_by_uuid([x[0] for x in cited_chunks])
    page_numbers = [x[2] for x in cited_chunks]

    for j, page_number in enumerate(page_numbers):
        if isinstance(page_number, str):
            page_numbers[j] = json.loads(page_number)

    response_markdown = replace_doc_ref(
        str(response["output_text"]),
        cited_files,
        page_numbers=page_numbers,
        flexible=True,
    )

    return response_markdown
