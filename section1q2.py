# Import necessary libraries
import pandas as pd  
import torch  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from inference import predict 

# Load your SGHateCheck data from CSV files
files = [  # List of file paths for the test case datasets
    'sghatecheckdata/ms_testcases_all.csv',
    'sghatecheckdata/ss_testcases_all.csv',
    'sghatecheckdata/ta_testcases_all.csv',
    'sghatecheckdata/zh_testcases_all.csv'
]

# Combine data from all CSV files into a single DataFrame
all_data = pd.concat([pd.read_csv(file) for file in files])  # Read and concatenate all data into one DataFrame

# Map true labels from text to numerical values (0 and 1)
label_mapping = {
    'hateful': 1,  # Map 'hateful' to 1
    'non-hateful': 0,  # Map 'non-hateful' to 0
}

# Convert true labels in the DataFrame to numerical format
all_data['true_labels'] = all_data['t_gold'].map(label_mapping)  # Apply mapping to 't_gold' column

# Prepare the input data for predictions
batch_text = all_data['c_testcase'].astype(str).tolist()  # Extract text from test cases

# Run evaluation on the combined dataset
predicted_results = predict(batch_text)  # Call the prediction function

# Extract predictions for all thresholds from the first category
first_category = next(iter(predicted_results.values()))  # Get the first category's results
predicted_high_recall = first_category['predictions']['high_recall']  # Extract high recall predictions
predicted_balanced = first_category['predictions']['balanced']  # Extract balanced predictions
predicted_high_precision = first_category['predictions']['high_precision']  # Extract high precision predictions

# Convert the predicted labels and true labels to tensors for evaluation
true_labels_tensor = torch.tensor(all_data['true_labels'].tolist())  # Convert true labels column to a tensor

# Evaluate High Recall predictions
predicted_high_recall_tensor = torch.tensor(predicted_high_recall)  # Convert high recall predictions to a tensor
accuracy_high_recall = accuracy_score(true_labels_tensor.numpy(), predicted_high_recall_tensor.numpy())
precision_high_recall = precision_score(true_labels_tensor.numpy(), predicted_high_recall_tensor.numpy())
recall_high_recall = recall_score(true_labels_tensor.numpy(), predicted_high_recall_tensor.numpy())
f1_high_recall = f1_score(true_labels_tensor.numpy(), predicted_high_recall_tensor.numpy())

# Evaluate Balanced predictions
predicted_balanced_tensor = torch.tensor(predicted_balanced)  # Convert balanced predictions to a tensor
accuracy_balanced = accuracy_score(true_labels_tensor.numpy(), predicted_balanced_tensor.numpy())
precision_balanced = precision_score(true_labels_tensor.numpy(), predicted_balanced_tensor.numpy())
recall_balanced = recall_score(true_labels_tensor.numpy(), predicted_balanced_tensor.numpy())
f1_balanced = f1_score(true_labels_tensor.numpy(), predicted_balanced_tensor.numpy())

# Evaluate High Precision predictions
predicted_high_precision_tensor = torch.tensor(predicted_high_precision)  # Convert high precision predictions to a tensor
accuracy_high_precision = accuracy_score(true_labels_tensor.numpy(), predicted_high_precision_tensor.numpy())
precision_high_precision = precision_score(true_labels_tensor.numpy(), predicted_high_precision_tensor.numpy())
recall_high_precision = recall_score(true_labels_tensor.numpy(), predicted_high_precision_tensor.numpy())
f1_high_precision = f1_score(true_labels_tensor.numpy(), predicted_high_precision_tensor.numpy())

# Print out the evaluation results for all thresholds
print("High Recall Predictions:")
print(f"Accuracy: {accuracy_high_recall:.4f}")  
print(f"Precision: {precision_high_recall:.4f}")  
print(f"Recall: {recall_high_recall:.4f}")  
print(f"F1 Score: {f1_high_recall:.4f}")  

print("\nBalanced Predictions:")
print(f"Accuracy: {accuracy_balanced:.4f}")  
print(f"Precision: {precision_balanced:.4f}")  
print(f"Recall: {recall_balanced:.4f}")  
print(f"F1 Score: {f1_balanced:.4f}")  

print("\nHigh Precision Predictions:")
print(f"Accuracy: {accuracy_high_precision:.4f}")  
print(f"Precision: {precision_high_precision:.4f}")  
print(f"Recall: {recall_high_precision:.4f}")  
print(f"F1 Score: {f1_high_precision:.4f}")  