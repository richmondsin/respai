# Section 2

## Q1: Off-topic prompt detection
Each component of the pipeline is chosen with the goal of optimizing model performance while ensuring the system remains efficient and scalable. Below, I will explain the motivation for selecting each component used in the model:

1. Dataset and Preprocessing:
    • Dropping rows with missing values ensures we only use clean, complete data for training, which is critical to avoid skewed or biased models. Removing missing values results in a high-quality training set, minimizing the risk of noise or incomplete data affecting performance.

2. Sampling for Efficient Prototyping:
    • I do not have the necessary GPUs to train the model with 2.6 million rows, and also very time consuming during experimentation and tuning. Sampling 10,000 rows enables faster iteration and prototyping, allowing me to test and refine the model before scaling to larger datasets.

3. Feature Engineering (Keyword Overlap and Prompt Length):
    • Keyword Overlap: This feature calculates how much overlap exists between words in the system_prompt and prompt. If many words overlap, it suggests relevance between the prompt and system prompt. This is a simple yet effective feature to detect common terms between the two texts.
    • Prompt Length: The length of the prompt can be an indicator of complexity. Longer prompts may include off-topic discussions, while shorter ones may be more straightforward and likely relevant. The inclusion of prompt length helps the model distinguish between complex and concise requests.

4. Semantic Embedding using DistilBERT and Cosine Similarity:
	• DistilBERT is a lightweight transformer model that is well-suited for embedding text with fewer resources compared to full-scale BERT. Using DistilBERT ensures that each system_prompt and prompt is represented as meaningful numerical vectors (embeddings), which can capture deeper semantic relationships beyond simple keyword overlap.
	• Cosine Similarity: This metric measures the similarity between the embeddings of the system_prompt and prompt. High similarity indicates that the user’s prompt is likely relevant to the system’s domain. Using cosine similarity as a feature allows us to capture the relationship between the meanings of the two inputs.

5. TF-IDF Vectorization for Combined Prompts:
	• TF-IDF (Term Frequency-Inverse Document Frequency) is a traditional method to quantify the importance of words in a document relative to a corpus. For text classification tasks, TF-IDF is a highly interpretable feature extraction method that balances word frequency and uniqueness, making it effective for distinguishing relevant and irrelevant prompts.
	• Combined Prompts: By concatenating system_prompt and prompt, the model can capture the joint context and vocabulary used in both. This step provides the model with a holistic view of the combined information, improving its ability to distinguish between on-topic and off-topic queries.

6. Model Selection:
	• Logistic Regression: A linear classifier that is often effective for text classification problems. It is fast to train, interpretable, and provides probabilistic outputs, making it ideal for a baseline model.
	• Random Forest: An ensemble method that builds multiple decision trees and aggregates their predictions. It excels at capturing non-linear relationships and works well with both numeric and categorical data. Random Forest is also resistant to overfitting, making it a good choice for classification tasks like this.
	• Support Vector Machine (SVM): SVM is a robust classifier that works well in high-dimensional spaces like the one created by TF-IDF and embeddings. It’s particularly effective in cases where the decision boundary between classes is clear but complex.

7. Hyperparameter Tuning with RandomizedSearchCV:
	• RandomizedSearchCV: Instead of exhaustively searching all combinations of hyperparameters (as done in GridSearch), RandomizedSearch samples random combinations. This approach is computationally cheaper and faster, especially for large datasets, while still providing optimal or near-optimal hyperparameters for each model.

8. Model Evaluation:
    • Accuracy: Measures the overall correctness of the model (i.e., how many times the model correctly identifies whether a prompt is on-topic or off-topic).
	• Precision: Ensures that when the model identifies a prompt as off-topic, it is correct. High precision is essential in avoiding false positives (on-topic queries incorrectly labeled as off-topic).
	• Recall: Measures the ability to correctly capture all off-topic queries. A higher recall means fewer off-topic prompts are missed.
	• F1-Score: A balance between precision and recall that is particularly useful for this problem, where there may be an imbalance between on-topic and off-topic prompts.
    • Cross-Validation: Using techniques like 3-fold cross-validation during hyperparameter tuning helps ensure the model is robust and can generalize well to unseen data.
	• Test Set Evaluation: Once the model is tuned, it’s important to test it on a held-out dataset (test set) to ensure that the performance metrics reflect real-world behavior.

Each component of the model-building process is chosen to balance interpretability, scalability, and performance. The combination of traditional techniques (TF-IDF and keyword overlap) and modern methods (DistilBERT embeddings) creates a robust feature set. The use of multiple classifiers ensures that we can compare model performances, and RandomizedSearchCV enables efficient hyperparameter tuning. This end-to-end pipeline provides an effective approach to off-topic prompt detection without relying on external LLM APIs.

Future Research and Improvements:

1. Larger Training Dataset:
	• With more compute power or time, expanding the dataset size would improve model generalization. Using datasets from diverse domains would help the tool adapt to various industries and topics.
2. Transformer-Based Models:
	• While we used DistilBERT for embedding, more powerful models like BERT-Large or RoBERTa could be explored if more resources were available. These models would capture more nuanced language patterns in the prompts.
3. Incremental Learning:
	• Implementing an incremental learning pipeline would allow the tool to continuously adapt based on feedback from product teams, incorporating new prompts as they come in, without retraining the model from scratch.
4. Advanced Interpretability:
	• Providing feature importance or attention-based interpretability for each prediction would help product teams understand why the model classified a prompt as off-topic, increasing transparency and trust in the system.