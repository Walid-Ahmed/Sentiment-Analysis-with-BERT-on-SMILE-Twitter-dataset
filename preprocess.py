from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import json

from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import numpy as np

def preprocess(data_path):
  df = pd.read_csv(data_path, names=['id', 'text', 'category'])
  # Drop column 'id'
  df = df.drop('id', axis=1)
  # Values to be removed
  values_to_remove = ["nocode",'happy|surprise', 'happy|sad',"disgust|angry","sad|disgust","sad|angry","sad|disgust|angry"]
  # Remove rows where column 'category' has values from values_to_remove
  df = df[~df['category'].isin(values_to_remove)]
  # Convert string values in 'category' to unique integer values
  df['label'],uniques = pd.factorize(df['category'])
  # Create a mapping from integer values to original category labels
  label_to_category = {index: label for index, label in enumerate(uniques)}
  # To get the reverse mapping from category labels to integer values
  category_to_label = {label: index for index, label in label_to_category.items()}
  # Get the number of unique values in the 'labels' column
  num_unique_values = df['label'].nunique()
  unique_values = df['category'].unique().tolist()
  X_train, X_val, y_train, y_val = train_test_split(
    df.text.values,
    df.label.values,
    test_size = 0.15,
    random_state = 17,
    stratify = df.label.values)
  return X_train, X_val, y_train, y_val,num_unique_values,unique_values,label_to_category

def tokenize(X_train,X_val):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
  encoded_data_train = tokenizer.batch_encode_plus(
      X_train,
      add_special_tokens = True,
      return_attention_mask = True,
      max_length = 256,
      padding='max_length',
      return_tensors = 'pt'
      ,truncation=True)

  encoded_data_val = tokenizer.batch_encode_plus(
    X_val,
    add_special_tokens = True,
    return_attention_mask = True,

    return_tensors = 'pt',
    padding='max_length',
    max_length = 256,)
  return encoded_data_train,encoded_data_val

def create_save_dataset(encoded_data_train,encoded_data_val,y_train,y_val,label_to_category):

  input_ids_train = encoded_data_train['input_ids']
  attention_masks_train = encoded_data_train['attention_mask']
  input_ids_val = encoded_data_val['input_ids']
  attention_masks_val = encoded_data_val['attention_mask']

  # Find unique elements
  unique_elements = np.unique(y_train)
  # Get the number of unique elements
  num_unique_values = len(unique_elements)


  labels_train = torch.tensor(y_train)
  labels_val = torch.tensor(y_val)
  dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
  dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

  # Save the datasets

  # Define the folder name
  folder_name = "data"

  # Check if the folder exists, if not, create it
  if not os.path.exists(folder_name):
      os.makedirs(folder_name)
      print("Folder 'data' created.")
  else:
      print("Folder 'data' already exists.")
  torch.save(dataset_train, 'data/dataset_train.pt')
  torch.save(dataset_val, 'data/dataset_val.pt')
  print("[INFO] Fils  dataset_train.pt and  dataset_val.pt saved to folder data")


  # Prepare the data to be stored in JSON format
  data_to_store = {
      'num_unique_values': num_unique_values,
      'unique_values': unique_values,
      'label_to_category':label_to_category
  }

  # Write the data to a JSON file
  with open('data_info.json', 'w') as json_file:
      json.dump(data_to_store, json_file, indent=4)

  print("Data Info stored in data_info.json successfully.")
  
def main():
    # The path to your CSV file
    csv_file_path = "smileannotationsfinal.csv"
    
    # Preprocess the data
    X_train, X_val, y_train, y_val = preprocess(csv_file_path)
    
    # Tokenize the data
    encoded_data_train, encoded_data_val = tokenize(X_train, X_val)
    
    # Create and save the dataset
   create_save_dataset(encoded_data_train, encoded_data_val)

if __name__ == "__main__":
   main()
