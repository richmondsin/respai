# Section 1

## Q1: Critiquing LionGuard
Read through our paper on LionGuard, “LionGuard: Building a Contextualized Moderation Classifier to Tackle Localized Unsafe Content”, which can be found [here](https://arxiv.org/abs/2407.10995).

### a. Why are low-resource languages challenging for natural language processing and large language models? Discuss this with reference to tokenisation.
There are 3 reasons why low-resource languages are challenging for NLP and large language models:
1. **Data scarcity**: Low-resource languages have limited data available for training NLP models. They may have numerous dialects and variations which are very different from the English language. This scarcity of data makes it difficult to build accurate and robust models for these languages. Large language models require a vast amount of data to learn the underlying patterns and relationships in the language, and the lack of data for low-resource languages hinders the performance of these models.

2. **Complex Morphology**: Many low-resource languages exhibit rich morphology, where a single word can take many forms based on tense, case, or gender. For example "I go lah" (I'm definitely going) versus "I go meh?" (Am I really going?). The tokenizer have to recognise these variations and their contextual usage to accurately process the sentence structure. This variability makes it harder to tokenize and require more sophisticated tokenization strategies to handle the morphological complexity of these languages. The lack of standardisation of the spelling, grammar and vocabulary in low-resource languages further complicates the tokenization process. 

3. **Cultural Context and Nuance**: Low-resource languages often have unique cultural contexts, idioms, and expressions that are not present in other languages. Words like "Kiasu", "Sibei", "Chope" have their own usage in Singapore but not in anywhere else. These cultural nuances can be challenging for large language models to understand and interpret correctly. For example, certain words or phrases may have different meanings in different cultural contexts, and large language models may struggle to capture these nuances without sufficient training data.

In the context of the LionGuard paper, these challenges highlight the importance of localized models that are specifically trained on low-resource languages.

### b. What implications does this paper have for the development of safety guardrails for LLMs in countries with low-resource languages?
The paper highlights the importance of developing safety guardrails for large language models (LLMs) in countries with low-resource languages. Here are some implications of the paper for the development of safety guardrails in such countries:

1. **Localization of Models**: The paper emphasizes that the existing LLMs used globally struggle with localised content. The guardrails should not solely rely on global models but should integrate localised models that are trained on data from the specific country or region to capture the unique linguistic characteristics, cultural nuances, and context-specific expressions of the language. 

2. **Importance of Robust Training Data**: To develop effective safety guardrails, it is crucial to create and curate large, high-quality datasets that reflect the diversity and complexity of the target language. The paper suggests that using automated labelling techniques, such as those implemented in LionGuard, can facilitate the generation of balanced training datasets, particularly when human annotation may be inconsistent.

3. **Collaboration with Local Experts**: Collaboration with local experts, linguists, and community members is essential for understanding the specific challenges and nuances of low-resource languages. By involving local stakeholders in the development and evaluation of safety guardrails, it can ensure that the models are culturally sensitive, contextually relevant, and effective in addressing local safety concerns.

### c. Why was PR-AUC used as the evaluation metric for LionGuard over other performance metrics? In what situations would a F1-score be preferable?
Precision-Recall Area Under the Curve (PR-AUC) is the metric that represents the area under the precision-recall curve. F1-score is the harmonic mean of precision and recall. 

There are several reasons why PR-AUC (Precision-Recall Area Under the Curve) was chosen as the evaluation metric for LionGuard over other performance metrics:

1. **Imbalanced Datasets**: PR-AUC is beneficial when dealing with imbalanced datasets, where the number of positive examples (unsafe content) is much smaller than the number of negative examples (safe content), which is the case for this paper. In this situation, metrics like accuracy could still be misleading because a model might achieve high accuracy simply by predicting the majority class (harmful content) more often without effectively identifying specific instances of non-harmful content or vice versa.

2. **Focus on Positive Class**: PR-AUC focuses on the precision and recall of the positive class, which is crucial for moderation tasks where the goal is to identify harmful or unsafe content accurately. It provides a more nuanced evaluation of the model's performance in detecting relevant instances of unsafe content while minimizing false positives.

3. **Trade-off between Precision and Recall**: PR-AUC captures the trade-off between precision and recall, allowing for a comprehensive assessment of the model's ability to balance between correctly identifying harmful content (recall) and avoiding false alarms (precision). This balance is particularly important in moderation tasks where both missing harmful content and incorrectly flagging safe content can have significant consequences. F1-score treats both classes equally, which may not be desirable in this context where one class is more important.

Cases where F1-score would be preferable over PR-AUC include scenarios where the class distribution is balanced, and both precision and recall are equally important. For instance, in tasks where false positives and false negatives have similar consequences, F1-score provides a single metric that balances precision and recall. However, in the context of LionGuard's moderation classifier, where the focus is on detecting harmful content with high precision and recall, PR-AUC is a more suitable metric.

### d. What are some weaknesses in LionGuard’s methodology?
There are several weaknesses in LionGuard's methodology that could be addressed to improve the model:

1. **Limited Exploration of Ambiguous Cases**: The paper highlights that AI can provide more consistent labeling compared to human annotators. However, this assertion carries risks. While AI can indeed reduce variability and inconsistency in labeling, it does not guarantee that the labels produced are accurate or contextually appropriate. AI models, including LionGuard, are trained on specific datasets that may not fully encompass the complexities of real-world language use, particularly for low-resource languages with rich morphological variations and localized expressions. If the AI model is trained on datasets that reflect biased or non-representative samples of language use, the resulting labels may reflect these biases consistently. This raises concerns about fairness and equity in moderation practices, as the AI may disproportionately mislabel content based on its training data rather than the actual intent or context of the language being used.

2. **Binary and Multi-Label Clasification**: The current categories of hateful, harassment, public harm, self-harm, sexual, toxic, and violent content capture significant types of harmful language, but they may not encompass the full spectrum of issues that arise in online interactions, particularly in the context of low-resource languages like Singlish. For example, in Singlish, certain phrases might convey sarcasm or social commentary that could be interpreted as harmful but don’t fit neatly into the established categories. The model might mislabel content that is culturally specific or contextually rich. For example, what might be a friendly jest among peers could be interpreted as harassment or toxic behavior by the model if it strictly adheres to predefined categories without understanding the context.

3. **Interpretability**: Interpretability is about understanding how a model makes decisions, which is crucial for trust in moderation systems. Users need to know why a particular piece of content was flagged or allowed to pass through moderation. Important aspects includes transparency in the model's decision-making process, the ability to trace back to the features that influenced the model's prediction, and the capacity to explain the model's reasoning in a human-understandable way. The PR-AUC does not account for how interpretable the model is, creating a black box in the decision-making processes. The use of autoencoders could possibly enhance interpretability through dimensionality reduction, reconstruction error analysis, and feature exploration, making it a potentially valuable tool in developing more trustworthy content moderation systems.

## Q2: Evaluating LionGuard
The experiment is done in the python file `section1q2.py`. The results based on different threshold are as follows:
- High Recall Predictions:
    - Accuracy: 0.5761
    - Precision: 0.7633
    - Recall: 0.5566
    - F1 Score: 0.6438

- Balanced Predictions:
    - Accuracy: 0.5121
    - Precision: 0.7950
    - Recall: 0.3922
    - F1 Score: 0.5252

- High Precision Predictions:
    - Accuracy: 0.4741
    - Precision: 0.8189
    - Recall: 0.3027
    - F1 Score: 0.4420

1.	High Recall Predictions:
- Accuracy: 57.61%: The accuracy is moderate, indicating LionGuard can correctly predict the labels more than half of the time in high-recall mode.
- Precision: 76.33%: This shows that when LionGuard predicts something as hateful, it is right 76.33% of the time. This is quite high, which is positive.
- Recall: 55.66%: The recall is lower, meaning it misses many actual hateful cases. However, for high-recall predictions, this suggests there may still be room for improvement in detecting more hateful content.
- F1 Score: 64.38%: The F1 score balances both precision and recall, and it suggests that LionGuard performs reasonably but could improve in identifying more hateful content while maintaining precision.

Conclusion: LionGuard tends to be cautious, prioritizing precision over recall, which limits its ability to catch all cases of hateful content but ensures that when it flags something, it is usually correct.

2.	Balanced Predictions:
- Accuracy: 51.21%: The accuracy drops here, reflecting a more neutral balance between precision and recall.
- Precision: 79.50%: Precision remains high, showing that LionGuard is still careful about flagging content, maintaining accuracy when it predicts hateful content.
- Recall: 39.22%: The recall drops significantly, meaning that LionGuard is missing a lot of actual hateful content.
- F1 Score: 52.52%: The F1 score shows a decline, signaling a struggle to balance precision and recall effectively.

Conclusion: When LionGuard balances precision and recall, it tends to prioritize precision too much, at the cost of recall, which results in it failing to detect many hateful cases.

3.	High Precision Predictions:
- Accuracy: 47.41%: Accuracy further decreases as the focus shifts to precision.
- Precision: 81.89%: This is the highest precision among the three settings, meaning LionGuard is very confident in its hateful predictions, rarely making mistakes in what it flags as hateful.
- Recall: 30.27%: However, recall is at its lowest, meaning LionGuard is missing a large majority of actual hateful content. It focuses on being highly precise, but sacrifices the ability to catch most hateful cases.
- F1 Score: 44.20%: The F1 score is the lowest, reflecting the poor balance between precision and recall.

Conclusion: In high-precision mode, LionGuard becomes very conservative, only flagging content when it’s extremely certain it’s hateful. However, this comes at the cost of recall, missing a lot of actual hateful content.

Summary:
- Where LionGuard excels: It does well in precision, especially in high-precision mode, where it ensures that flagged content is indeed hateful.
- Where LionGuard struggles: It struggles in recall across all modes, especially in high-precision and balanced settings. This limits its ability to catch all hateful content, leading to a significant number of false negatives.

Improvements should focus on enhancing recall, especially without drastically sacrificing precision. This would allow LionGuard to identify a larger portion of hateful content while maintaining its current precision.

## Q3: Improving LionGuard
I have some ideas on how LionGuard can be improved:

1. **Leveraging Sparse Autoencoders for Feature Extraction**: Sparse autoencoders can be used to extract meaningful features from the text data, capturing the essential information that contributed to the decision making process. By incorporating sparse autoencoders into the model architecture, LionGuard can enhance its ability to identify subtle patterns and nuances in the language, particularly in low-resource languages with complex morphology and cultural context that resulted in the classification of harmful content.

    - Input for the Sparse Autoencoder (SAE)
        - Input Data: Numerical representation of the text data from the low-resource language, such as word embeddings. Each input example (e.g., a comment or a sentence) is represented as a high-dimensional vector.

    - Training the Sparse Autoencoder
        - Structure:
            - Input Layer: Matches the dimension of the input feature vectors as defined by the embedding.
            - Hidden Layer: Contains fewer neurons than the input layer, encouraging the model to compress the input data.
            - Output Layer: Matches the dimension of the input layer.

        - Training Process:
            - The SAE is trained to minimize the reconstruction error, meaning it learns to reproduce its input at the output layer. The sparsity constraint encourages the model to focus on essential features.

    - Output from the Sparse Autoencoder
        - Encoded Features:
            - The output of the hidden layer (the compressed representation) serves as the new feature set for each input example. This representation captures the significant patterns in the data while filtering out noise.

    - Using the Encoded Features with the Ridge Classifier
        - Input to Ridge Classifier:
            - Once the SAE is trained, you pass the original feature vectors (text embeddings) through the encoder part of the SAE to obtain the compressed representations.
            - These encoded features become the input for the Ridge Classifier.
        - Training the Ridge Classifier:
            - The Ridge Classifier is then trained using the compressed features as input and the corresponding labels (e.g., toxic, non-toxic) as the target output.
            - The classifier will learn to map the lower-dimensional representations (encoded features) to the respective classes.

This process leverages the strengths of both models: the SAE focuses on extracting meaningful features, while the Ridge Classifier effectively makes predictions based on those features. This combination can lead to improved accuracy and interpretability in tasks such as detecting hate speech or other forms of toxic language.

2. **Feedback Loop**: Implement a continuous learning framework that allows LionGuard to adapt to new data over time. This can help it stay relevant as language evolves, especially in a diverse linguistic environment like Singapore where new slang and expressions emerge frequently.

    - Data Feedback Loop:
        - Incorporate a mechanism to collect feedback from users regarding the moderation decisions made by LionGuard. This feedback can be used to refine the model continuously.
        - Annotate using human or AI to review flagged content and correct any misclassifications. This data can then be fed back into the training process through a database for feedback.

    - Incremental Training:
        - Implement a system for incremental training for the Ridge classifier, where LionGuard can be updated periodically with new data without requiring a complete retraining from scratch. This can include fine-tuning the model on recent examples of toxic language or new forms of hate speech.

By adopting this approach, LionGuard can remain relevant and effective in detecting harmful content in a changing linguistic landscape in Singapore.