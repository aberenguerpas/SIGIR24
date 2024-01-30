import argparse
import faiss
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def retrieve_data(index):
    """
    Retrieve all embeddings from a given index
    """
    ids = np.arange(index.ntotal).astype('int64')
    base_embs = []

    for id in ids:
        base_embs.append(index.reconstruct_n(int(id), 1)[0])

    return np.array(base_embs)


def main():
    parser = argparse.ArgumentParser(description='Build a general index')
    parser.add_argument('-i', '--input', default='../exp1/embeddings/dublin/',
                        help='Embeddings index directory')
    parser.add_argument('-m', '--model', default='all',
                        choices=['all', 'uae-large',
                                 'bge-large', 'bge-base',
                                 'gte-large', 'ember'])
    args = parser.parse_args()

    dataset = args.input.split("/")[-2]
    print(dataset)

    files = os.listdir(args.input)

    models = []
    if args.model == 'all':
        models = ['uae-large', 'bge-large', 'bge-base', 'gte-large', 'ember']
    else:
        models.append(args.model)

    for m in models:
        dimensions = 1024

        if m == 'bge-base':
            dimensions = 768

        # Index to save the embeddings
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dimensions))

        # Dataset index id
        id = 0

        # Dataframe to map (id - dataset)
        map = pd.DataFrame()

        for file in tqdm(files):
            if file.split("_")[0] == m:
                try:
                    # Load table index
                    index_tab = faiss.read_index(args.input + file)

                    # Retrieve table embeddings and normalize
                    embs = retrieve_data(index_tab)
                    embs = np.array([np.mean(embs, axis=0)])
                    faiss.normalize_L2(embs)

                    # Index the average embeddins a general index
                    # and map (id-dataset)
                    index.add_with_ids(embs, np.array([id]))
                    new_row = {"id": id, "dataset": file}
                    map = pd.concat([map, pd.DataFrame([new_row])],
                                    ignore_index=True)
                    id += 1
                except Exception:
                    pass

        if not os.path.isdir('./index_files'):
            os.mkdir('./index_files')

        # Save index and mapping
        faiss.write_index(index, "./index_files/"+m+"_"+dataset+".index")
        map.to_csv("./index_files/"+m+"_"+dataset+"_map.csv", index=False)


if __name__ == "__main__":
    main()
