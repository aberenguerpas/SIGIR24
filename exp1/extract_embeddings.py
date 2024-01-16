import argparse
import faiss
import os
import pandas as pd
import numpy as np
import traceback
from tqdm import tqdm
from utils import get_model, find_delimiter

# Convert text to embedding
def enconde_text(model_name, model, text):
    if model_name == 'uae-large':
        return model.encode(text, to_numpy=True)
    else:
        return [model.encode(text, show_progress_bar=False)]
    
# Extract embeddings from each row
def content_embeddings(model, df, size, model_name):
    all_embs = np.empty((0, size), dtype=np.float32)

    for _, row in df.iterrows():

        text = " ".join(map(str, row.values.flatten().tolist()))
        # Create embedding from chunks
        embs = enconde_text(model_name, model, text)

        if len(embs) > 1:
            print(len(embs))

        all_embs = np.append(all_embs, embs, axis=0)

    return all_embs

def main():
    parser = argparse.ArgumentParser(description='Process Darta')
    parser.add_argument('-i', '--input', default='sensors',
                        choices=['sensors', 'wikitables', 'chicago'],
                        help='Directorio de los datos')
    parser.add_argument('-m', '--model', default='all',
                        choices=['all', 'uae-large', 'bge-large', 'bge-base', 'gte-large', 'ember'])
    parser.add_argument('-t', '--type', default='baseline',
                        choices=['baseline', 'random', 'cluster'])
    parser.add_argument('-r', '--result', default='./indexs',
                        help='Name of the output folder that stores the indexs files')
    
    args = parser.parse_args()

    dataset = args.input
    args.input = '../data/' + args.input + '/'
    files = os.listdir(args.input)

    models = []
    if args.model == 'all':
        models = ['uae-large', 'bge-large', 'bge-base', 'gte-large', 'ember']
    else:
        models.append(args.model)

    for m in models:
        model, dimensions = get_model(m)
        model.max_seq_length = dimensions

        # Saves key ("m" + "_" + dataset + "_" + "file") - value (embeddings)
        # map = pd.DataFrame()

        for file in tqdm(files):
            try:
                # Read dataframe
                delimiter = find_delimiter(args.input + file)
                df = pd.read_csv(args.input + file, sep=delimiter, nrows=100)

                # Remove columns with all NaNs
                df = df.dropna(axis='columns', how='all')
                df.dropna(how='all', inplace=True)

                # Calculate embeddings
                embs = content_embeddings(model, df, dimensions, m)

                # Normalize them
                faiss.normalize_L2(embs)

                # Configure Faiss index
                dim = embs.shape[1]             # Embeddings dimensions
                index = faiss.IndexFlatL2(dim)  # Euclidian L2 Index

                # Add embeddings to index
                index.add(embs)

                # Save index in file
                faiss.write_index(index, 'embeddings/' + dataset + '/' + m + '_' + dataset + '_' + file.replace('.csv', '') + '.index')
                
            except Exception as e:
                print('Error en archivo', file)
                print(e)
                print(traceback.format_exc())

if __name__ == "__main__":
    main()