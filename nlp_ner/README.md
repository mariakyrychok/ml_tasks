# Mountain Entity Recognition (NER)

## Project Overview

This project focuses on training a Named Entity Recognition (NER) model to identify mountain names within textual data. 
The repository includes all necessary scripts, datasets, and documentation to facilitate the training and 
inference of the model.

## Dataset Creation
The dataset consists of three main components, each created using different approaches:

1. `top_100_mountain_names.csv`:

- This file contains a list of the top 100 most popular mountains.
- The data was gathered from open data sources that provide information on well-known mountains around the world.

2. `sentences.csv`:
- This file includes raw text sentences and mountain names.
- To generate this dataset, I utilized the OpenAI API to produce diverse and natural-sounding examples. 
You can create a new dataset by running the `generate_data.py` script. 
However, I recommend using the pre-generated dataset for consistency. 
If you choose to run the script (`python3 generate_data.py`), ensure you have an `OPENAI_API_KEY`. 
For access to the key or further details, feel free to contact me.

3. `labeled_sentences.csv`:
- These are labeled examples derived from the raw sentences in `sentences.csv`, 
with mountain names tagged as entities for training the NER model.

## Model Training
1. The dataset was tokenized using the `BertTokenizerFast`, and token labels were aligned based on 
a custom tag-to-ID mapping scheme. 
2. The model was fine-tuned on the annotated dataset using the `Trainer` class from the Hugging Face Transformers library.
3. The `seqeval` metric was employed for model evaluation, enabling fine-tuning by measuring the accuracy of 
token classification across entities like mountains.

## Model inference
1. The input text is first split into tokens using nltk's word tokenizer. 
2. The tokens are then passed to the BERT tokenizer for further processing, ensuring they are properly formatted for the model. 
3. Once the tokens are passed through the model, predicted labels are returned, and these are mapped back to their 
corresponding entity tags, distinguishing between entities like B-MOUNTAIN and I-MOUNTAIN.

## Link to model weight
You can find model weights [here](https://drive.google.com/file/d/1IivChAtiC23vvPdRIJ6r6m4NVD5KS9hR/view?usp=sharing)

## Usage
For usage run `demo.ipynb`