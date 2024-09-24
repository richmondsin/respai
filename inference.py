import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
import sys
import json
import onnxruntime as rt

# Download model config
repo_path = "govtech/lionguard-v1"
config_path = hf_hub_download(repo_id=repo_path, filename="config.json")
with open(config_path, 'r') as f:
    config = json.load(f)

def get_embeddings(device, data):

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['embedding']['tokenizer'])
    model = AutoModel.from_pretrained(config['embedding']['model'])
    model.eval()
    model.to(device)

    # Generate the embeddings
    batch_size = config['embedding']['batch_size']
    num_batches = int(np.ceil(len(data)/batch_size))
    output = []
    for i in range(num_batches):
        sentences = data[i*batch_size:(i+1)*batch_size]
        encoded_input = tokenizer(sentences, max_length=config['embedding']['max_length'], padding=True, truncation=True, return_tensors='pt')
        encoded_input.to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        output.extend(sentence_embeddings.cpu().numpy())
    
    return np.array(output)

def predict(batch_text):

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    embeddings = get_embeddings(device, batch_text)
    embeddings_df = pd.DataFrame(embeddings)

    # Prepare input data
    X_input = np.array(embeddings_df, dtype=np.float32)

    # Load the classifiers
    results = {}
    for category, details in config['classifier'].items():

        # Download the classifier from HuggingFace hub
        local_model_fp = hf_hub_download(repo_id = repo_path, filename = config['classifier'][category]['model_fp'])

        # Run the inference
        session = rt.InferenceSession(local_model_fp)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: X_input})

        # If calibrated, return only the prediction for the unsafe class
        if config['classifier'][category]['calibrated']: 
            scores = [output[1] for output in outputs[1]]

        # If not calibrated, we will only get a 1D array for the unsafe class
        else:
            scores = outputs[1].flatten()
        
        # Generate the predictions depending on the recommended threshold score
        results[category] = {
            'scores': scores,
            'predictions': {
                'high_recall': [1 if score >= config['classifier'][category]['threshold']['high_recall'] else 0 for score in scores],
                'balanced': [1 if score >= config['classifier'][category]['threshold']['balanced'] else 0 for score in scores],
                'high_precision': [1 if score >= config['classifier'][category]['threshold']['high_precision'] else 0 for score in scores]
            }
        }

    return results

if __name__ == "__main__":

    # Load the data
    input_data = sys.argv[1]
    batch_text = json.loads(input_data)

    # Generate the scores and predictions
    results = predict(batch_text)
    for i in range(len(batch_text)):
        print(f"Text: '{batch_text[i]}'")
        for category in results.keys():
            print(f"[Text {i+1}] {category} score: {results[category]['scores'][i]:.3f} | HR: {results[category]['predictions']['high_recall'][i]}, B: {results[category]['predictions']['balanced'][i]}, HP: {results[category]['predictions']['high_precision'][i]}")
        print('---------------------------------------------')