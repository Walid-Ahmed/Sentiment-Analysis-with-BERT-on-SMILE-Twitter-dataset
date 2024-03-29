

# BERT Sentiment Analysis on SMILE Twitter Dataset

This repository contains the implementation of a sentiment analysis model using the BERT (Bidirectional Encoder Representations from Transformers) architecture, fine-tuned on the SMILE Twitter dataset. The SMILE Twitter dataset, curated by Wang et al., encompasses a variety of tweets labeled with emotions, providing a rich source for sentiment analysis tasks.


BERT is designed to include a special token, [CLS], at the beginning of each input sequence. During pre-training, BERT learns to use this token for classification tasks.
For sentence-level classification tasks (e.g., sentiment analysis), the embedding of the [CLS] token, after processing by BERT, can be used as the aggregate representation of the sentence. This embedding is then fed into a classification layer to make predictions.
The [CLS] token is explicitly trained to capture the context useful for classification tasks, making it a strong choice for these types of problems


## Dataset

The SMILE Twitter Emotion dataset was created by Wang, Bo; Tsakalidis, Adam; Liakata, Maria; Zubiaga, Arkaitz; Procter, Rob; Jensen, Eric in 2016. It includes tweets annotated with emotions such as happiness, anger, sadness, and others. For further details and to download the dataset, visit the [SMILE Twitter Emotion dataset page](https://doi.org/10.6084/m9.figshare.3187909.v2).

![Sample 5 rows](figs/sample_data.png)

## Installation

To run this project, ensure you have Python 3.6+ installed. Then, install the required packages using pip:

```
pip install -r requirements.txt
```

This command will install all necessary libraries, including `transformers`, `torch`, and others required for data processing and model training.

## Usage

### Data Preprocessing

First, preprocess the SMILE Twitter dataset to convert it into a format suitable for BERT training:

```
python preprocess.py 
```
This will save 2 files in data folder:   dataset_train.pt and dataset_val.pt with aNother file data_info.json that looks as follow:

![Data Info](figs/data_info.png)

### Model Training

To fine-tune the BERT model on the preprocessed dataset, run:

```
python train.py 
```

### Evaluation

Evaluate the model's performance on the evaluation  dataset:

```
python evaluate.py 
```

## Model

We utilize the `bert-base-uncased` model from the `transformers` library as our starting point, fine-tuning it on the emotion-labeled tweets. The fine-tuning process adjusts the pre-trained BERT model to better perform sentiment analysis specific to the SMILE dataset.

![Model](figs/model.png)


Certainly! Here's a concise version of the `TextDataset` class summary for inclusion in your GitHub README file:

---

## TextDataset Class Overview

The `TextDataset` class is a custom PyTorch dataset for preparing text data for NLP models like BERT. It handles tokenization, padding, truncation, and attention mask generation, streamlining the process of converting raw text into model-ready formats. Optionally, it can also manage labels for supervised tasks.

**Key Features:**
- Utilizes Hugging Face's `BertTokenizer` for efficient tokenization.
- Automatically pads or truncates texts to a uniform length.
- Generates attention masks to differentiate data from padding.
- Supports inclusion of labels for each text input.

**Usage:**

```python
dataset = TextDataset(texts=["Sample text"], labels=[0])
dataloader = DataLoader(dataset, batch_size=1)
```

**Requirements:** PyTorch, Hugging Face `transformers`.

This class simplifies data preprocessing for training and evaluating transformer-based NLP models.

--- 

This summary provides a clear, brief overview of what the `TextDataset` class does, how to use it, and its dependencies, making it suitable for a quick reference in your project's README file.

## Results

After fine-tuning, the model achieves a notable accuracy in classifying tweets into their respective emotion categories. Detailed results and analysis will be found in the `results` folder.

![Training Loss](figs/Training_Loss.png)
![Confusion Matrix](figs/confusion_matrix.png)

##  Inference from saved model

```
tweet = "I hate this movie"
label = predict_label(tweet)
print(f"Predicted label: {label}")
```

## Contribution

Contributions to this project are welcome. Please follow the existing code style and submit pull requests for any enhancements, bug fixes, or documentation improvements.

## Citation

If you use the SMILE Twitter dataset or this sentiment analysis model in your research, please cite the original dataset as follows:

```
@misc{wang2016smile,
  author = {Wang, Bo and Tsakalidis, Adam and Liakata, Maria and Zubiaga, Arkaitz and Procter, Rob and Jensen, Eric},
  title = {SMILE Twitter Emotion dataset},
  year = {2016},
  publisher = {figshare},
  doi = {10.6084/m9.figshare.3187909.v2}
}
```

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.

---


