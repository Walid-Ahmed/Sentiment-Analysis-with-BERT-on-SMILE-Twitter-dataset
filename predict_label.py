from transformers import BertTokenizer, BertForSequenceClassification
import torch
import json

def index_to_label(prediction_index,label_to_category):
    # Define your mapping from index to label
    label_dict = label_to_category
        # Add more labels as needed

    # Return the corresponding label for the prediction index
    return label_dict.get(str(prediction_index), "Unknown")  # Default to "Unknown" if index is not found



# Function to predict the label of a tweet
def predict_label(tweet):

  # Load the data from the JSON file
  with open('data_info.json', 'r') as json_file:
      loaded_data = json.load(json_file)

  # Extract values into variables
  num_unique_values = loaded_data['num_unique_values']
  unique_values = loaded_data['unique_values']
  label_to_category=loaded_data['label_to_category']
  

  # Print the loaded data to verify
  print(f"Number of Unique Values: {num_unique_values}")
  print(f"Unique Values: {unique_values}")
  print(f"label_to_category: {label_to_category}")

  # Check if CUDA is available and set the device accordingly
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print(f"Using device: {device}")
  state_dict= torch.load('Bert_ft.pt',
              map_location = device)
  model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=num_unique_values,
                                                      output_attentions=False,
                                                      output_hidden_states=False)
  model.load_state_dict(state_dict)
  # Ensure the model is in evaluation mode
  model.eval()

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
  # Tokenize and encode the tweet for BERT
  inputs = tokenizer.encode_plus(
        tweet,
        None,
        add_special_tokens=True,
        max_length=256,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

  # Get the input IDs and attention mask tensors
  input_ids = inputs['input_ids']
  attention_mask = inputs['attention_mask']

  # Make prediction
  with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

  # Get the prediction (the index of the highest logit)
  prediction = torch.argmax(outputs.logits, dim=1).item()


  # Convert prediction index to label (assuming you have a way to map indices to labels)
  label = index_to_label(prediction,label_to_category)  # Implement this function based on your labels

  return label

if __name__ == "__main__":
    # Example usage
    tweet = "I hate this movie"
    label = predict_label(tweet)
    print(f"for Tweet\"{tweet}\" Predicted label: {label}")
