from datasets import load_dataset
from tqdm import tqdm
import pandas as pd  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import joblib

# Load the dataset
ds = load_dataset("gabrielchua/off-topic")

# Convert the dataset to a pandas DataFrame
train_df = pd.DataFrame(ds['train'])

# Step 1: Remove rows with missing values
initial_rows = train_df.shape[0]
train_df_cleaned = train_df.dropna(subset=['system_prompt', 'prompt'])
final_rows = train_df_cleaned.shape[0]
rows_removed = initial_rows - final_rows

print(f"Number of rows before removing missing values: {initial_rows}")
print(f"Number of rows after removing missing values: {final_rows}")
print(f"Number of rows removed: {rows_removed}")

# Randomly sample 10,000 rows for faster prototyping
train_df_cleaned = train_df_cleaned.sample(n=10000, random_state=42)

# Step 2: Combine system_prompt and prompt
train_df_cleaned['combined_prompt'] = train_df_cleaned['system_prompt'] + " " + train_df_cleaned['prompt']

# Step 3: Feature Engineering - Adding Keyword Overlap and Length of Prompt
# Enable tqdm progress bar for pandas apply
tqdm.pandas()

# Function to calculate keyword overlap between system_prompt and prompt
def keyword_overlap(system_prompt, prompt):
    system_keywords = set(system_prompt.split())
    prompt_words = set(prompt.split())
    return len(system_keywords.intersection(prompt_words))

# Apply keyword overlap with progress bar
print("Calculating keyword overlap...")
train_df_cleaned['keyword_overlap'] = train_df_cleaned.progress_apply(lambda x: keyword_overlap(x['system_prompt'], x['prompt']), axis=1)

# Apply prompt length calculation with progress bar
print("Calculating prompt length...")
train_df_cleaned['prompt_length'] = train_df_cleaned['prompt'].progress_apply(lambda x: len(x.split()))

# Step 4. Semantic Embedding using DistilBERT and Cosine Similarity
# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Function to generate DistilBERT embeddings for a given text
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Add tqdm progress bar for system_prompt and prompt embeddings
tqdm.pandas()  # Enable progress bar for pandas apply

# Generate embeddings with progress bar for system_prompt and prompt
print("Generating embeddings for system_prompt...")
train_df_cleaned['system_prompt_embedding'] = train_df_cleaned['system_prompt'].progress_apply(get_embedding)

print("Generating embeddings for prompt...")
train_df_cleaned['prompt_embedding'] = train_df_cleaned['prompt'].progress_apply(get_embedding)

# Add tqdm progress bar for cosine similarity calculation
print("Calculating cosine similarity between system_prompt and prompt...")
train_df_cleaned['similarity'] = tqdm(train_df_cleaned.apply(
    lambda x: cosine_similarity(x['system_prompt_embedding'], x['prompt_embedding'])[0][0], axis=1), 
    total=len(train_df_cleaned)
)

# Step 4. Prepare Final Dataset for Training
# Combine the features: TF-IDF of combined_prompt, keyword_overlap, prompt_length, similarity

# Step 4.1: Vectorize the combined prompt using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)
X_tfidf = vectorizer.fit_transform(train_df_cleaned['combined_prompt'])

# Convert the sparse TF-IDF matrix to a DataFrame
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

# Step 4.2: Combine the TF-IDF features with keyword overlap, prompt length, and similarity
additional_features = train_df_cleaned[['keyword_overlap', 'prompt_length', 'similarity']].reset_index(drop=True)
X_features = pd.concat([tfidf_df.reset_index(drop=True), additional_features], axis=1)

# Target variable
y = train_df_cleaned['off_topic']

# Step 5. Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_features, y, test_size=0.3, random_state=42)  # Split 70% train, 30% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Split 50-50 from temp to val and test

# Print the sizes of the splits
print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Step 6. Hyperparameter Tuning using GridSearchCV on Validation Set

# Define parameter distributions for Logistic Regression and Random Forest
param_distributions = {
    "Logistic Regression": {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200]
    },
    "Random Forest": {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    }
}

# Initialize classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()  # Default parameters for SVM (no hyperparameter tuning)
}

# Initialize variables to track the best model across all classifiers
best_overall_model = None
best_overall_score = 0
best_model_name = ''
best_model_params = None

# Use GridSearchCV to find the best hyperparameters for Logistic Regression and Random Forest
for name, clf in classifiers.items():
    if name in ["Logistic Regression", "Random Forest"]:
        print(f"Starting hyperparameter tuning for {name}...")

        # Perform hyperparameter tuning for Logistic Regression and Random Forest
        param_dist = param_distributions[name]
        random_search = RandomizedSearchCV(
            estimator=clf,
            param_distributions=param_dist,
            n_iter=10,
            scoring='f1',
            n_jobs=-1,
            cv=3,
            verbose=2,
            random_state=42
        )
        random_search.fit(X_train, y_train)
        print(f"Best hyperparameters for {name}: {random_search.best_params_}")
        best_model = random_search.best_estimator_

    else:
        # Use default SVM without hyperparameter tuning
        clf.fit(X_train, y_train)
        best_model = clf
        print(f"Using default SVM without tuning for {name}")

    # Evaluate the model on the validation set
    y_val_pred = best_model.predict(X_val)
    
    # Evaluate performance on validation set
    accuracy_val = accuracy_score(y_val, y_val_pred)
    precision_val = precision_score(y_val, y_val_pred)
    recall_val = recall_score(y_val, y_val_pred)
    f1_val = f1_score(y_val, y_val_pred)

    print(f"Validation Results for {name}:")
    print(f"Accuracy: {accuracy_val:.4f}")
    print(f"Precision: {precision_val:.4f}")
    print(f"Recall: {recall_val:.4f}")
    print(f"F1 Score: {f1_val:.4f}\n")

    # Keep track of the best model based on the F1 score on the validation set
    if f1_val > best_overall_score:
        best_overall_score = f1_val
        best_overall_model = best_model
        best_model_name = name
        best_model_params = random_search.best_params_ if name in ["Logistic Regression", "Random Forest"] else "Default SVM parameters"

    # Evaluate the model on the test set
    y_test_pred = best_model.predict(X_test)

    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_test = recall_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred)

    print(f"Test Results for {name}:")
    print(f"Accuracy: {accuracy_test:.4f}")
    print(f"Precision: {precision_test:.4f}")
    print(f"Recall: {recall_test:.4f}")
    print(f"F1 Score: {f1_test:.4f}\n")

# Save the best model and vectorizer
print(f"Saving the best model ({best_model_name}) with parameters {best_model_params}...")
joblib.dump(best_overall_model, 'off_topic_detector/models/off_topic_model.pkl')

print("Saving the TF-IDF vectorizer...")
joblib.dump(vectorizer, 'off_topic_detector/models/tfidf_vectorizer.pkl')

# Print out the details of the best model
print("\nBest Overall Model Details:")
print(f"Best Model Name: {best_model_name}")
print(f"Best Model F1 Score: {best_overall_score:.4f}")
print(f"Best Model Parameters: {best_model_params}")

print("Model and vectorizer saved successfully!")