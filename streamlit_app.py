pip install --upgrade pip

import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re
import time

#Side Bar
st.sidebar.subheader("About the app")
st.sidebar.info("Scrape and Summarize CNN News Indonesia with Streamlit")
st.sidebar.write("\n\n")
st.sidebar.markdown("Use it carefully")
st.sidebar.divider()
st.sidebar.write("Make sure the link you take comes from the CNN Indonesia page")
st.sidebar.write("\n\n")
st.sidebar.caption("Created by Ungga Putra using [Streamlit](https://streamlit.io/)🎈.")


def scrap_and_summarize(link):
    # Scrap berita
    cnn_url = link
    html = requests.get(cnn_url)
    bsobj = BeautifulSoup(html.content, 'html.parser')
    
    headlines = [link.text for link in bsobj.findAll("h1")]
    news_text = [re.sub(r'\n', '', news.text.strip()) for news in bsobj.findAll('div', {'class': 'detail-text text-cnn_black text-sm grow min-w-0'})]
    
    # Menampilkan hasil scraping
    st.header("Headlines:")
    for headline in headlines:
        st.write(headline)
    
    st.header("News:")
    for news in news_text:
        st.write(news)
    
    
    # Ringkaskan berita
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
    news_input = ' '.join(news_text)  # Menggabungkan seluruh teks berita
    summary = summarizer(news_input, max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Menampilkan hasil ringkasan
    st.header("News Summary:")
    st.write(summary[0]['summary_text'])

# Tampilan Streamlit
st.title("Scrape and Summarize CNN News with Streamlit")

# Input URL dari pengguna
url_input = st.text_input("Input CNN news URL:")

# Ketika tombol "Scrape dan Ringkaskan" ditekan
if st.button("Scrape and Summarize"):
    if url_input:
        # Panggil fungsi scrap_and_summarize dengan URL sebagai parameter
        with st.spinner('Doing some AI magic, please wait...'):
            time.sleep(2)
            scrap_and_summarize(url_input)
    else:
        st.warning("Enter the news URL first!")
