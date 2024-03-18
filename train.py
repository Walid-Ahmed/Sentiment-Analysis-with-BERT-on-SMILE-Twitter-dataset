


def train():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  num_unique_values=6

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
  batch_size = 4 #32


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
          batch = tuple(t.to(device) for t in batch)
          inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2]}

          model.zero_grad()

          outputs = model(**inputs)
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
  plt.show()


if __name__ == "__main__":
   train()

