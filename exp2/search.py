import argparse
import faiss
import os
import pandas as pd
import numpy as np
import sklearn
import traceback
from tqdm import tqdm
from utils import get_model, find_delimiter


# Convertir texto a embedding
def enconde_text(model_name, model, text):
    if model_name == 'uae-large':
        return model.encode(text, to_numpy=True)
    else:
        return [model.encode(text, show_progress_bar=False)]


# Saca embeddings de cada fila
def content_embeddings(model, df, size, model_name):
    # Array vacio donde meter todos y luego promediarlos
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
                        choices=['sensors', 'wikitables', 'chicago', 'dublin'],
                        help='Directorio de los datos')
    parser.add_argument('-m', '--model', default='all',
                        choices=['all', 'uae-large', 'bge-large',
                                 'bge-base', 'gte-large', 'ember'])
    parser.add_argument('-r', '--result', default='./indexs',
                        help='Output folder for indexs files')

    args = parser.parse_args()

    dataset = args.input

    files = []
    if dataset == 'wikitables':
        args.input = './benchmarks_wikitables/queries.txt'
        files = open(args.input, "r").readlines()
    else:
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

        index = faiss.read_index("./index_files/"+m+"_"+dataset+".index")
        map = pd.read_csv("./index_files/"+m+"_"+dataset+"_map.csv")

        results = pd.DataFrame(columns=['q', 'P@1', 'RR'])
        print(files)
        for ir, file in enumerate(tqdm(files)):
            try:
                if dataset == 'wikitables':
                    file_name = file.split("\t")[1].strip()
                    df = pd.read_csv('../data/wikitables/' + file_name + ".csv")

                else:
                    # Read dataframe
                    delimiter = find_delimiter(args.input + file)
                    df = pd.read_csv(args.input + file, sep=delimiter)

                    # Remove columns with all NaNs
                    df = df.dropna(axis='columns', how='all')
                    df.dropna(how='all', inplace=True)

                    # Se mezcla
                    df = sklearn.utils.shuffle(df, random_state=0)

                    # Seleccionamos % de la tabla que sera indexado
                    ratio = 0.10
                    total_rows = df.shape[0]
                    search_size = int(total_rows*ratio)

                    # Datos en para indexar
                    df_search = df[-search_size:]

                    # Se calculan los embeddings
                    embs = content_embeddings(model, df_search, dimensions, m)
                    embs = np.array([np.mean(embs, axis=0)])
                    # Se normalizan
                    faiss.normalize_L2(embs)

                    distances, ann = index.search(np.array(embs), k=10)
                    #results_aux = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})

                    # Calculate P_1
                    r_id = ann[0][0]
                    p_1 = 0
                    print(map.iloc[r_id]['dataset'], file)
                    if map.iloc[r_id]['dataset'] == file:
                        p_1 = 1

                    score_rr = 0
                    for k, r in enumerate(ann[0]):
                        if map.iloc[r_id]['dataset'] == file:
                            score_rr = 1/(k+1)
                            break
                    # add result to results dataframe
                    results = results._append({'q': file,
                                    'P@1': p_1,
                                    'RR': score_rr},
                                    ignore_index = True)
                #print(results.head())
            except Exception as e:
                print(e)
        results.to_csv('./results/'+m+'_'+dataset+'.csv')


if __name__ == "__main__":
    main()
