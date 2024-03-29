from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch

from torch.utils.data import Dataset
from transformers import BertTokenizer

class TextDataset(Dataset):
    """A custom Dataset class for tokenizing text data for BERT, including labels."""
    
    def __init__(self, texts, labels, max_length=512):
        """
        Args:
            texts (list of str): The list of sentences to tokenize.
            labels (list of int): The labels for each sentence.
            max_length (int): The maximum length of a sequence after tokenization.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Tokenizes a sentence at the specified index in the dataset and returns
        its input IDs, attention mask, and label.
        """
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'  # Return PyTorch tensors
        )
        
        input_ids = inputs['input_ids'].squeeze()  # Remove batch dimension
        attention_mask = inputs['attention_mask'].squeeze()  # Remove batch dimension
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label)  # Ensure label is a tensor
        }


'''
# Example usage:
texts = ["Hello, how are you?", "I'm fine, thank you!"]
labels = [0, 1]  # Example binary labels for demonstration

dataset = TextDataset(texts, labels)

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch in dataloader:
    print(batch['input_ids'], batch['attention_mask'], batch['labels'])

'''    
