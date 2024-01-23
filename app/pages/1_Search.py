import pathlib

import streamlit as st
from tqdm import tqdm
from utils import add_chunks_to_vector_store, init_session_state

from ofstedai.api.ofsted_api import (
    BASE_OFSTED_URL,
    extract_reports,
    extract_school_pages,
    get_pages,
)
from ofstedai.models.file import Chunk, File
from ofstedai.parsing.file_chunker import FileChunker

init_session_state()


def get_reports_from_url(url: str):
    with st.spinner("Fetching pages from Ofsted..."):
        pages = get_pages(url)

    with st.spinner("Fetching pages from Ofsted..."):
        school_page_progress_bar = st.progress(0)
        for i, page in tqdm(
            enumerate(pages),
            desc="Fetching links: ",
            unit="page",
            position=0,
            leave=True,
        ):
            schools = extract_school_pages(page)
            school_page_progress_bar.progress(float(i + 1) / len(pages))
        # Hide the progress bar
        school_page_progress_bar.empty()

    reports_and_schools = []
    with st.spinner("Fetching reports from Ofsted..."):
        school_report_progress_bar = st.progress(0)
        for i, school_url in tqdm(
            enumerate(schools),
            desc="Fetching school reports: ",
            unit="school",
            position=0,
            leave=True,
        ):
            extracted_report_paths, school_names = extract_reports(school_url)
            for i, report in enumerate(extracted_report_paths):
                reports_and_schools.append((report, school_url, school_names[i]))
            school_report_progress_bar.progress(float(i + 1) / len(schools))
        # Hide the progress bar
        school_report_progress_bar.empty()

    return reports_and_schools


file_chunker = FileChunker()


st.title("Ofsted AI Copilot - Search 🔍")

st.markdown(
    """Use the search to populate the AI Copilot with Ofsted reports to talk about."""
)

# UI Components for the search form

search_input = st.text_input("Name, URN or keyword")
location_input = st.text_input("Location or postcode")

if location_input:
    distance_options_miles = [3, 5, 10]
    distance_input = st.select_slider(
        "Distance (Miles from Location)", options=distance_options_miles
    )

categories = {
    "All categories": None,
    "Education and training": 1,
    "Childcare and early education": 2,
    "Children’s social care": 3,
    "Other organisations": 4,
}

categories_input = st.selectbox("Category", list(categories.keys()))


# Build the search URL with the inputs

url_input = f"{BASE_OFSTED_URL}/search?q={search_input}"
if location_input:
    url_input += "&location=" + location_input
    if distance_input:
        url_input += f"&radius={distance_input}"

if categories[categories_input] is not None:
    url_input += f"&level_1_types={categories[categories_input]}"

run_button = st.button("Load School Reports")

if run_button:
    reports_and_schools = get_reports_from_url(url_input)

    if len(reports_and_schools) == 0:
        st.warning("⚠️ No reports found. Try broadening your search criteria.")
        st.stop()

    report_indexing_progress_bar = st.progress(0)
    for i, report_and_school in enumerate(reports_and_schools):
        report_path, school_url, school_name = report_and_school
        file_type = pathlib.Path(report_path).suffix
        file_name = pathlib.Path(report_path).stem

        file = File(
            path=report_path,
            school_url=school_url,
            school_name=school_name,
            type=file_type,
            name=file_name,
            storage_kind="local",
            creator_user_uuid="dev",
        )

        with st.spinner(f"Chunking **{file.name}**"):
            try:
                chunks = file_chunker.chunk_file(file=file)
            except TypeError as err:
                st.error(f"Failed to process {file.name}, error: {str(err)}")
                raise err

        # ==================== SAVING ====================

        with st.spinner(f"Saving **{file.name}**"):
            st.session_state.storage_handler.write_item(item=file)
            st.session_state.storage_handler.write_items(items=chunks)

        # ==================== INDEXING ====================

        with st.spinner(f"Indexing **{file.name}**"):
            add_chunks_to_vector_store(chunks=chunks)

        st.toast(body=f"{file.name} Complete")
        report_indexing_progress_bar.progress(float(i + 1) / len(reports_and_schools))
    report_indexing_progress_bar.empty()

    st.success(f"✅ Successfully loaded reports")
