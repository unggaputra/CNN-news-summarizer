import re
import time
import requests
from bs4 import BeautifulSoup
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="CNN News Summarizer", layout="centered")

# Sidebar
st.sidebar.subheader("About the app")
st.sidebar.info("Scrape and Summarize CNN News Indonesia with Streamlit")
st.sidebar.markdown("Use it carefully")
st.sidebar.divider()
st.sidebar.write("Make sure the link you take comes from the CNN Indonesia page")
st.sidebar.caption("Created by Ungga Putra using [Streamlit](https://streamlit.io/)🎈.")


@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn",
    )


def scrap_and_summarize(link: str):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(link, headers=headers, timeout=15)
    response.raise_for_status()

    bsobj = BeautifulSoup(response.content, "html.parser")

    headlines = [h1.get_text(strip=True) for h1 in bsobj.find_all("h1") if h1.get_text(strip=True)]
    detail_divs = bsobj.find_all("div", class_=re.compile(r"\bdetail-text\b"))
    news_text = [
        re.sub(r"\s+", " ", div.get_text(" ", strip=True))
        for div in detail_divs
        if div.get_text(strip=True)
    ]

    st.header("Headlines:")
    if headlines:
        for headline in headlines:
            st.write(headline)
    else:
        st.info("No headline (`<h1>`) found.")

    st.header("News:")
    if not news_text:
        st.warning("No article body found. The page structure may have changed or the URL is not a news detail page.")
        return

    for paragraph in news_text:
        st.write(paragraph)

    news_input = " ".join(news_text).strip()
    if len(news_input.split()) < 30:
        st.info("Article text is too short to summarize effectively.")
        return

    summarizer = load_summarizer()
    summary = summarizer(
        news_input,
        max_length=200,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        truncation=True,
        do_sample=False,
    )

    st.header("News Summary:")
    st.write(summary[0]["summary_text"])


st.title("Scrape and Summarize CNN News with Streamlit")
url_input = st.text_input("Input CNN news URL:")

if st.button("Scrape and Summarize"):
    if url_input.strip():
        try:
            with st.spinner("Doing some AI magic, please wait..."):
                time.sleep(1)
                scrap_and_summarize(url_input.strip())
        except requests.RequestException as e:
            st.error(f"Failed to fetch URL: {e}")
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
    else:
        st.warning("Enter the news URL first!")
