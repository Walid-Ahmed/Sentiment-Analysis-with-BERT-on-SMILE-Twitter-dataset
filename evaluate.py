from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import torch
import json
import seaborn as sns
import os



def evalModel():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  num_unique_values=6
  path_to_model="Bert_ft.pt"


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


  # Load fine-tuned-model 
  model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=num_unique_values,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

  
  state_dict= torch.load('Bert_ft.pt',
              map_location = device)
  model.load_state_dict(state_dict)

  model.eval()  # Put the model in evaluation mode
  model.to(device)


  dataset_val=torch.load('data/dataset_val.pt')

  dataloader_validation = DataLoader(
      dataset_val,
      sampler = SequentialSampler(dataset_val),
      batch_size = 32
  )

  predictions, true_labels = [], []

  with torch.no_grad():
      for batch in dataloader_validation:
          batch = tuple(b.to(device) for b in batch)
          inputs = {
              'input_ids'       : batch[0],
              'attention_mask'  : batch[1],
              'labels'          : batch[2]
          }

          outputs = model(**inputs)

          # Move logits and labels to CPU as they will be converted to numpy
          logits = outputs.logits.detach().cpu().numpy()
          label_ids = batch[2].to('cpu').numpy()

          # Store predictions and true labels
          predictions.append(logits)
          true_labels.append(label_ids)

  # Convert the predictions and labels to flat lists
  flat_predictions = np.concatenate(predictions, axis=0)
  flat_true_labels = np.concatenate(true_labels, axis=0)

  # For classification, we can use the argmax to get the predicted label
  flat_predictions = np.argmax(flat_predictions, axis=1)

  # Calculate the accuracy
  accuracy = accuracy_score(flat_true_labels, flat_predictions)
  print(f"Accuracy: {accuracy}")
  plot_save_confusion_matrix(unique_values,flat_true_labels, flat_predictions)

def plot_save_confusion_matrix(unique_values,flat_true_labels, flat_predictions):
  class_labels = unique_values

  # Compute the confusion matrix
  conf_matrix = confusion_matrix(flat_true_labels, flat_predictions)

  # Optionally, normalize the confusion matrix
  conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

  # Plotting the confusion matrix
  plt.figure(figsize=(10, 8))
  sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues",
              xticklabels=class_labels,  # Set custom x-tick labels
              yticklabels=class_labels)  # Set custom y-tick labels
  plt.ylabel('Actual')
  plt.xlabel('Predicted')
  plt.title('Normalized Confusion Matrix')


  # Define the folder name
  folder_name = "results"

  # Check if the folder exists, if not, create it
  if not os.path.exists(folder_name):
      os.makedirs(folder_name)
      print("Folder results created.")
  else:
      print("Folder results already exists.")
  plt.savefig(os.path.join("results", 'confusion_matrix.png'))
  plt.show()



if __name__ == "__main__":
   evalModel()
