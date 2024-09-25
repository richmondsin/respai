import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import os

# Load the model and vectorizer
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models', 'off_topic_model.pkl')
vectorizer_path = os.path.join(current_dir, 'models', 'tfidf_vectorizer.pkl')

best_model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Initialize tokenizer and model for embeddings
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def keyword_overlap(system_prompt, prompt):
    system_keywords = set(system_prompt.split())
    prompt_words = set(prompt.split())
    return len(system_keywords.intersection(prompt_words))

def is_prompt_relevant(system_prompt, user_prompt):
    # Feature Engineering
    keyword_overlap_value = keyword_overlap(system_prompt, user_prompt)
    prompt_length = len(user_prompt.split())

    # Generate embeddings
    system_prompt_embedding = get_embedding(system_prompt)
    prompt_embedding = get_embedding(user_prompt)
    similarity = cosine_similarity(system_prompt_embedding, prompt_embedding)[0][0]

    # TF-IDF Vectorization
    combined_input = system_prompt + " " + user_prompt
    combined_tfidf = vectorizer.transform([combined_input])
    combined_tfidf_df = pd.DataFrame(combined_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

    # Combine features
    features = pd.concat([
        combined_tfidf_df.reset_index(drop=True),
        pd.DataFrame({
            'keyword_overlap': [keyword_overlap_value],
            'prompt_length': [prompt_length],
            'similarity': [similarity]
        })
    ], axis=1)

    # Ensure feature columns match training data
    features = features[best_model.feature_names_in_]

    # Prediction
    prediction = best_model.predict(features)

    # If prediction is 0, it means relevant (True), if 1 it means off-topic (False)
    return prediction[0] == 0