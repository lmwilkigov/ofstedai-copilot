import os
import time
from functools import lru_cache

import requests
import streamlit as st
import typer
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_OFSTED_URL = "https://reports.ofsted.gov.uk"
data = {"pages": [], "schools": []}

if not os.path.exists("./data"):
    os.makedirs("./data")

if not os.path.exists("./data/Ingest"):
    os.makedirs("./data/Ingest")


@st.cache_data
def make_request(url, max_retries=5, delay_seconds=5):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)

        if response.status_code == 403:
            print(
                f"Received 403 Forbidden. Retrying... (Attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(delay_seconds)
        elif response.status_code == 200:
            return response
        else:
            # Handle other status codes if needed
            print(f"Received unexpected status code: {response.status_code}")
            break

    print(f"Failed to retrieve data after {max_retries} attempts. Exiting.")
    return None


@st.cache_data
def extract_next_page_url(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    next_button = soup.find("a", class_="pagination__next")

    if next_button:
        return BASE_OFSTED_URL + next_button["href"]


@st.cache_data
def get_pages(url):
    data["pages"].append(url)
    found_all_pages = False
    with tqdm(
        total=None, desc="Fetching page links: ", unit="page", position=0, leave=True
    ) as pbar:
        while not found_all_pages:
            next_page_url = extract_next_page_url(url)
            if next_page_url:
                data["pages"].append(next_page_url)
                url = next_page_url
                pbar.update(1)  # Increment the progress bar
            else:
                found_all_pages = True
    return data["pages"]


@st.cache_data
def extract_school_pages(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    links = soup.select("ul.results-list > li > h3")
    for link in links:
        tag = link.find("a", href=True)
        if tag:
            data["schools"].append(BASE_OFSTED_URL + tag["href"])
    return data["schools"]


@st.cache_data
def extract_reports(url):
    page = make_request(url)
    soup = BeautifulSoup(page.content, "html.parser")
    report_paths = []
    school_names = []
    school = soup.find("h1", class_="heading--title").text.strip()
    timeline_ol = soup.find("ol", class_="timeline")
    for li in timeline_ol.find_all("li", class_="timeline__day"):
        # Skip the events with class 'timeline__day--opened'
        if "timeline__day--opened" in li.get("class", []):
            continue

        try:
            # Find the publication link within the <a> tag
            publication_link = li.find("a", class_="publication-link")
            date_span = publication_link.find("span", class_="nonvisual")

            if not date_span:
                raise Exception("No date span found for given element")

            date_text = date_span.text.strip()

            inspection_type = date_text.split(",")[0].strip().replace(" ", "-")
            formatted_date = date_text.split("-")[-1].strip()
            formatted_date = formatted_date.lower().replace(" ", "-")

            result_string = f"{inspection_type.lower()}_{formatted_date}"

            pdf_url = publication_link["href"]

            filename = f"{school}_{result_string}.pdf"
            filename = filename.replace(" ", "-").lower()

            response = make_request(pdf_url)
            if response:
                with open("./data/Ingest/" + filename, "wb") as pdf_file:
                    pdf_file.write(response.content)
                    report_paths.append("./data/Ingest/" + filename)
                    school_names.append(school)
        except:
            continue
    return report_paths, school_names
