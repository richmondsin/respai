# Off-Topic Detector

The **Off-Topic Detector** is a machine learning-based tool designed to filter off-topic prompts for LLM-powered chatbots. It detects whether a user's prompt is relevant to a given system prompt (the chatbot's domain), helping to improve resource utilization and accuracy by avoiding irrelevant queries.

## Key Features

- **Off-topic Detection**: The model determines whether a user prompt is relevant to a system prompt, acting as a guardrail before the prompt is passed to an LLM.
- **Semantic Embeddings**: The tool leverages semantic embeddings and other machine learning techniques to assess the relevance of user inputs.
- **Customizable Domains**: Users can define the domain for the chatbot, and the tool evaluates prompts against this domain.
- **Efficient Filtering**: Provides a more efficient solution to handle off-topic queries than relying solely on an LLM API.

## Document Structure

off_topic_detector/
│
├── __init__.py                     # Initializes the package (can be left empty or used for package-level imports)
│
├── detector.py                     # Contains the main logic for the off-topic detection
│   └── is_prompt_relevant()        # Function to check if the user prompt is relevant to the system prompt
│
├── models/                         # Directory for storing trained models and associated vectorizers
│   ├── off_topic_model.pkl         # Trained classification model (Random Forest, Logistic Regression, etc.)
│   ├── tfidf_vectorizer.pkl        # TF-IDF vectorizer used during feature extraction
│
├── setup.py                        # Setup script to install the package and its dependencies

## Installation

To install the Off-Topic Detector:

1. Activate your virtual environment (recommended).

## Training the Model

Before you can use the tool, you need to train the model locally. You can use the provided scripts:

- [section2q1.py](../section2q1.py): Python script for training the off-topic detection model.
- [section2q1.ipynb](../section2q1.ipynb): Jupyter notebook for training the off-topic detection model.

Ensure that you have installed all necessary dependencies before running the scripts.

### Option 1: Using Python Script (`section2q1.py`)

Run the following command to train the model:
    ```bash
    python section2q1.py
    ```

### Option 2: Using Jupyter Notebook (section2q1.ipynb)

Run the following command to train the model:

1.	Open the notebook with:

    ```bash
    jupyter notebook section2q1.ipynb
    ```

2.	Run all cells to train the model and save the trained model and vectorizer to the same directory as above.

## Model Storage

- The trained model is saved in off_topic_detector/models/off_topic_model.pkl.
- The corresponding TF-IDF vectorizer is saved in off_topic_detector/models/tfidf_vectorizer.pkl.

## Using the Off-Topic Detector

- Once the model is trained, you can use the tool to detect whether a user prompt is relevant to a chatbot’s domain. Here’s an example:
    
```python
from off_topic_detector.detector import is_prompt_relevant

system_prompt = "You are a gardening assistant who helps users plan and maintain their home gardens. You provide advice on plant selection, watering schedules, and soil maintenance for different types of plants."
user_prompt = "What is the best watering schedule for indoor succulents?"

result = is_prompt_relevant(system_prompt, user_prompt)
print("Is the prompt relevant?", result)
# Expected Output: True
```

- Other test cases to evaluate the tool's performance and accuracy in detecting off-topic prompts:

Sample 1:
```python
system_prompt = "You are a language learning assistant, helping users practice and improve their language skills through vocabulary, grammar exercises, and conversation practice."
user_prompt = "Can you help me practice conjugating French verbs?"
result = is_prompt_relevant(system_prompt, user_prompt)

print("Is the prompt relevant?", result)
# Expected Output: True
```

Sample 2:
```python
system_prompt = "You are an online banking assistant. You help users manage their bank accounts, transfer funds, and provide advice on financial products and services."
user_prompt = "What’s the best strategy to beat the final boss in Dark Souls?"
result = is_prompt_relevant(system_prompt, user_prompt)

print("Is the prompt relevant?", result)
# Expected Output: False
```

- system_prompt: This is the predefined context or domain of the chatbot (e.g., financial advisor).
- user_prompt: This is the query submitted by the user.
- is_prompt_relevant: This function checks if the user_prompt is relevant to the system_prompt. The function returns True if relevant, and False otherwise.

