import numpy as np
import faiss
import argparse
import os
import pandas as pd


def recover_data(index):
    ids = np.arange(index.ntotal).astype('int64')
    base_embs = []

    for id in ids:
        base_embs.append(index.reconstruct_n(int(id), 1)[0])

    return np.array(base_embs)


def main():
    parser = argparse.ArgumentParser(description='Process Darta')
    parser.add_argument('-i', '--input', default='sensors',
                        choices=['sensors', 'wikitables', 'chicago'],
                        help='Directorio de los datos')
    parser.add_argument('-m', '--model', default='all',
                        choices=['all', 'uae-large', 'bge-large', 'bge-base', 'gte-large', 'ember'])

    args = parser.parse_args()

    dataset = args.input
    args.input = '../exp1/embeddings/' + args.input + '/'

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

        # Index to save the whole embeddings
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dimensions))
        # Id del dataset en el indice
        id = 0
        # Dataframe que guarda los emparejamiento de id - archivo
        map = pd.DataFrame()

        for file in files:
            # Load original index of embeddings
            index = faiss.read_index(args.input + file)

            # Recover original embeddings
            embs = recover_data(index)
            embs = np.array([np.mean(embs, axis=0)])
            # Se normalizan
            faiss.normalize_L2(embs)

            # Se indexan y se las pasa un id
            index.add_with_ids(embs, np.array([id]))
            new_row = {"id": id, "dataset": file}
            map = pd.concat([map, pd.DataFrame([new_row])],
                            ignore_index=True)
            id += 1

        faiss.write_index(index, "./index_files/"+m+"_"+dataset+".index")
        map.to_csv("./index_files/"+m+"_"+dataset+"_map.csv", index=False)

if __name__ == "__main__":
    main()