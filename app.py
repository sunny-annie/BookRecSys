import streamlit as st
import pandas as pd
from functions import recommend_books_sbert_with_explanations

# Streamlit интерфейс
st.set_page_config(page_title="Система книжных рекомендаций", layout="wide")
st.title("📚 Система книжных рекомендаций")

query = st.text_input("Что бы вы хотели почитать?", placeholder="Например: нестрашная уютная детективная история")

if query:
    with st.spinner("Ищем книги..."):
        recommendations = recommend_books_sbert_with_explanations(query)

    for book in recommendations:
        cols = st.columns([1, 3])
        with cols[0]:
            st.image(book['image_url'], width=200)
            
        with cols[1]:
            st.subheader(book['название'])
            st.markdown(f"**Автор:** {book['автор']}")
            st.markdown(f"**Описание:** {book['описание']}")
            st.markdown(f"**Почему эта книга может вам подойти:** {book['объяснение']}")
            st.markdown(f"**Подходит на:** {int(book['сходство']*100)}%")
            st.markdown("---")
