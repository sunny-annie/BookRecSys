import pandas as pd
import re
import csv
import os
import torch
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# Hugging Face API
HF_TOKEN = os.getenv('HF_TOKEN')  
client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=HF_TOKEN)

# Загрузка модели и данных
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
df = pd.read_csv("full_df.csv", sep='*', escapechar='\\', quoting=csv.QUOTE_NONE) 
corpus_embeddings = torch.load("corpus_embeddings.pt")

# постобработка ответа LLM
def postprocess_llm_response(response):
    text = re.sub(r'<.*?>', '', response) # удаляем HTML-теги
    text = text.strip()

    # обрезаем ответ на последнем полном предложении
    if '.' in text:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        full_text = ''
        for sent in sentences:
            full_text += sent + ' '
            if sent.endswith(('.', '!', '?')):
                break
        return full_text.strip()
    return text

# функция создания промта для LLM
def explain_book_choice(query, book):
    prompt = f"""
Ты — помощник рекомендательной системы книг. Пользователь задал запрос: "{query}".

Вот описание подходящей книги:
Название: {book['название']}
Автор: {book['автор']}
Описание: {book['описание']}

Объясни, почему эта книга хорошо соответствует запросу. Ответь на русском языке, коротко и ясно.
"""

    response = client.text_generation(
        prompt=prompt,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9
    )

    return postprocess_llm_response(response)

# функция построения рекомендаций
def recommend_books_sbert_with_explanations(query, top_n=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_n)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        idx = int(idx)
        book = {
            'название': df.iloc[idx]['title'],
            'автор': df.iloc[idx]['author'],
            'описание': df.iloc[idx]['annotation'],
            'image_url': df.iloc[idx].get('image_url', None)
        }
        explanation = explain_book_choice(query, book)
        results.append({
            **book,
            'сходство': score.item(),
            'объяснение': explanation
        })
    return results