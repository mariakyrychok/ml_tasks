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

3. `annotated_sentences.csv`:
- These are labeled examples derived from the raw sentences in `sentences.csv`, 
with mountain names tagged as entities for training the NER model.