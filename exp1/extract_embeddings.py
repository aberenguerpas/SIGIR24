# import sklearn
import faiss
import pandas as pd
import numpy as np
import traceback
import csv
from sklearn.utils import shuffle
from tqdm import tqdm
from utils import get_model, find_delimiter
from sklearn.metrics.pairwise import cosine_similarity

# Convert text to embedding
def enconde_text(model_name, model, text):
    if model_name == 'uae-large':
        return model.encode(text, to_numpy=True)
    else:
        return [model.encode(text, show_progress_bar=False)]
    
# Extract embeddings from each row
def content_embeddings(model, df, size, model_name, tokenizer):
    all_embs = np.empty((0, size), dtype=np.float32)

    for _, row in df.iterrows():
        text = " ".join(map(str, row.values.flatten().tolist()))
        
        batch_dict = tokenizer(text,  max_length=512, return_attention_mask=False, padding=False, truncation=True)
        # Filter that the row has no more than 512 tokens
        if len(batch_dict['input_ids']) < 512:
            # Create embedding from chunks
            embs = enconde_text(model_name, model, text)
            if len(embs) > 1:
                print(len(embs))

            all_embs = np.append(all_embs, embs, axis=0)

    return all_embs

def recover_data(index):
    ids = np.arange(index.ntotal).astype('int64')
    base_embs = []

    for id in ids:
        base_embs.append(index.reconstruct_n(int(id), 1)[0])
    
    return np.array(base_embs)

# Extract base embeddings from the datasets
def extract_base_embeddings(args, dataset, files, models):
    for m in models:
        model, tokenizer, dimensions = get_model(m)
        model.max_seq_length = dimensions

        # Saves key ("m" + "_" + dataset + "_" + "file") - value (embeddings)
        # map = pd.DataFrame()

        for file in tqdm(files):
            try:
                # Read dataframe
                delimiter = find_delimiter(args.input + file)
                df = pd.read_csv(args.input + file, sep=delimiter)

                # Remove columns with all NaNs
                df = df.dropna(axis='columns', how='all')
                df.dropna(how='all', inplace=True)

                # Calculate embeddings
                embs = content_embeddings(model, df, dimensions, m, tokenizer)
                ids = np.array(range(0, len(embs))).astype(np.int64)

                # Normalize them
                faiss.normalize_L2(embs)

                # Configure Faiss index
                dim = embs.shape[1]             # Embeddings dimensions
                index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

                # Add embeddings to index
                index.add_with_ids(embs, ids)

                # Save index in file
                faiss.write_index(index, 'embeddings/' + dataset + '/' + m + '_' + dataset + '_' + file.replace('.csv', '') + '.index')
                
            except Exception as e:
                print('Error en archivo', file)
                print(e)
                print(traceback.format_exc())

def extract_random_reording_embeddings(args, dataset, files, models):
    for m in models:
        model, tokenizer, dimensions = get_model(m)
        model.max_seq_length = dimensions
        avg_similarities = np.array(0)
        std_similarities = np.array(0)

        for file in tqdm(files):
            try:
                # Read dataframe
                delimiter = find_delimiter(args.input + file)
                df = pd.read_csv(args.input + file, sep=delimiter, nrows=100)

                # Remove columns with all NaNs
                df = df.dropna(axis='columns', how='all')
                df.dropna(how='all', inplace=True)

                # Reorder the columns randomly
                df = shuffle(df)

                # Calculate embeddings
                embs = content_embeddings(model, df, dimensions, m, tokenizer)

                # Load original index of embeddings
                index = faiss.read_index('embeddings/' + dataset + '/' + m + '_' + dataset + '_' + file.replace('.csv', '') + '.index')

                # Recover original embeddings
                base_embs = recover_data(index)
                print(base_embs)
                print(embs)

                # Compare original embbedings with embeddings obtained after having mixed the table
                similarity_scores = cosine_similarity(base_embs, embs)

                # Average value of embeddings similarities
                avg_similarity = np.mean(similarity_scores)

                # Standard deviation of embeddings similarities
                std_similarity = np.std(similarity_scores)

                # Save these values
                avg_similarities = np.append(avg_similarities, avg_similarity)
                std_similarities = np.append(std_similarities, std_similarity)
            except Exception as e:
                print('Error en archivo', file)
                print(e)
                print(traceback.format_exc())
        
        # Average value of model similarities
        avg_similarity = np.mean(avg_similarities)

        # Average value of the standard deviations of the model similarities
        std_similarity = np.std(std_similarities)

        fields = np.array(["Average, Standard desviation"])
        values = np.append(avg_similarity, std_similarity)

        # Write values in a CSV file
        file_name = 'results/random_reordering/' + dataset + '/' + m + '_' + dataset + '.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            writer.writerow(values)