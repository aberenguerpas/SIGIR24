import faiss
import pandas as pd
import numpy as np
import traceback
from tqdm import tqdm
from utils import get_model, find_delimiter, content_embeddings

# Extract base embeddings from the datasets
def extract_base_embeddings(args, dataset, files, models):
    for m in models:
        model, tokenizer, dimensions = get_model(m)
        model.max_seq_length = 512

        # Saves key ("m" + "_" + dataset + "_" + "file") - value (embeddings)
        # map = pd.DataFrame()

        for file in tqdm(files[:500]):
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
                faiss.write_index(index, '/app/raid/embeddings/' + dataset + '/' + m + '_' + dataset + '_' + file.replace('.csv', '') + '.index')
                
            except Exception as e:
                print('Error en archivo', file)
                print(e)
                print(traceback.format_exc())