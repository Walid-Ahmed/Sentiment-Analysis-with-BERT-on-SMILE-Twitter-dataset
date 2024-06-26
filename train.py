import torch
from transformers import BertForSequenceClassification
from transformers import  get_linear_schedule_with_warmup
import tqdm
from tqdm import tqdm  # Import the tqdm function
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import json
import os
def train():

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




  # Load the data from the JSON file
  with open('data_info.json', 'r') as json_file:
      loaded_data = json.load(json_file)

  # Extract values into variables
  num_unique_values = loaded_data['num_unique_values']
  unique_values = loaded_data['unique_values']

  # Print the loaded data to verify
  print(f"Number of Unique Values: {num_unique_values}")
  print(f"Unique Values: {unique_values}")

  model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = num_unique_values,
    output_attentions = False,
    output_hidden_states = False)
  model.to(device)
  
  print("-"*70)
  print(model)
  print("-"*70)


  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 1e-5, #2e-5 to 5e-5
    eps = 1e-8)

  epochs=10
  batch_size = 4


  dataset_train=torch.load('data/dataset_train.pt')
  dataset_val=torch.load('data/dataset_val.pt')

  dataloader_train = DataLoader(
      dataset_train,
      sampler = RandomSampler(dataset_train),
      batch_size = batch_size
  )

  dataloader_validation = DataLoader(
      dataset_val,
      sampler = SequentialSampler(dataset_val),
      batch_size = 32
  )

    # Store the average loss after each epoch so we can plot them.
  loss_values = []

  for epoch in range(1, epochs+1):
      model.train()
      total_loss = 0

      for step, batch in enumerate(tqdm(dataloader_train, desc=f"Epoch {epoch}")):
        
          # Move each component of the batch to the GPU
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          labels = batch['labels'].to(device)  # Only if you're doing supervised learning
    

          model.zero_grad()
        
          outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

          loss = outputs[0]


        
          total_loss += loss.item()

          loss.backward()
          optimizer.step()

      avg_train_loss = total_loss / len(dataloader_train)
      loss_values.append(avg_train_loss)

      print(f"Epoch {epoch} | Average Training Loss: {avg_train_loss}")

  #save model
  torch.save(model.state_dict(), f'Bert_ft.pt')
  print("[INFO] Model saved to file  Bert_ft.pt")
  plot(loss_values,epochs)
  
def plot(loss_values,epochs):
  # Plotting the training loss
  plt.figure(figsize=(10, 6))
  plt.plot(loss_values, 'b-o')

  plt.title("Training Loss-BERT Sentiment Analysis ")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.xticks(range(1, epochs+1))


    # Define the folder name
  folder_name = "results"

  # Check if the folder exists, if not, create it
  if not os.path.exists(folder_name):
      os.makedirs(folder_name)
      print("Folder results created.")
  else:
      print("Folder results already exists.")
  plt.savefig(os.path.join("results", 'Training_Loss.png'))
  print("[INFO] Training Loss saved to file Training_Loss.png in results folder")
  plt.show()


if __name__ == "__main__":
   train()

