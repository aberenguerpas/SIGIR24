import argparse
import faiss
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import get_model


def save_result(id, rank, path_result):
    """
    Save search results in a suitable format to trec_eval
    """
    df = pd.DataFrame.from_dict(rank)
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'rank'})
    df['query_id'] = id
    df['Q0'] = 'Q0'
    df['STANDARD'] = 'STANDARD'
    df['score'] = df['score'].round(3)

    # Sort columns
    new_cols = ["query_id",
                "Q0", "document_id",
                "rank", "score", "STANDARD"]
    df = df[new_cols]
    df = df.reindex(columns=new_cols)

    df.to_csv(path_result, mode='a', index=False, header=False, sep="\t")


def enconde_text(model_name, model, text):
    """
    Convert text to embedding
    """
    if model_name == 'uae-large':
        return model.encode(text, to_numpy=True)
    else:
        return [model.encode(text, show_progress_bar=False)]


def content_embeddings(model, df, size, model_name):
    """
    Get a embedding for each dataset row
    """
    all_embs = np.empty((0, size), dtype=np.float32)

    for _, row in df.iterrows():
        text = " ".join(map(str, row.values.flatten().tolist()))
        embs = enconde_text(model_name, model, text)
        all_embs = np.append(all_embs, embs, axis=0)

    return all_embs


def main():
    parser = argparse.ArgumentParser(description='Search in the indexed data')
    parser.add_argument('-i', '--input',
                        default='./benchmarks_wikitables/queries.txt',
                        help='Queries file')
    parser.add_argument('-d', '--dataset', default='wikitables',
                        help='Dataset name')
    parser.add_argument('-m', '--model', default='all',
                        choices=['all', 'uae-large', 'bge-large',
                                 'bge-base', 'gte-large', 'ember'])
    parser.add_argument('-r', '--result', default='./search_results/',
                        help='Output folder')
    args = parser.parse_args()

    if not os.path.isdir(args.result):
        os.mkdir(args.result)

    models = []
    if args.model == 'all':
        models = ['uae-large', 'bge-large', 'bge-base', 'gte-large', 'ember']
    else:
        models.append(args.model)

    for m in models:
        model, dimensions = get_model(m)
        model.max_seq_length = dimensions

        index = faiss.read_index("./index_files/"+m+"_"+args.dataset+".index")
        map = pd.read_csv("./index_files/"+m+"_"+args.dataset+"_map.csv")

        files = open(args.input, "r").readlines()
        for file in tqdm(files):
            try:
                file_name = file.split("\t")[1].strip()
                id = file.split("\t")[0].strip()
                df = pd.read_csv('../data/'+args.dataset+'/'+file_name+".csv")

                # Calculate and normalize embeddings
                embs = content_embeddings(model, df, dimensions, m)
                embs = np.array([np.mean(embs, axis=0)])
                faiss.normalize_L2(embs)

                # Search in the index
                distances, ann = index.search(np.array(embs), k=20)

                # Format and save the results
                documents_ids = []
                for i in ann[0]:
                    idx = map.iloc[i]['dataset'].split(".")[0].split("_")[-1]
                    documents_ids.append(idx)

                rank = pd.DataFrame({'document_id': documents_ids,
                                     'score': distances[0]})

                save_result(id, rank, args.result+m+'_'+args.dataset+'.csv')

            except Exception:
                pass


if __name__ == "__main__":
    main()
