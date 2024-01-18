import argparse
import faiss
import os
import pandas as pd
import numpy as np
import sklearn
import traceback
from tqdm import tqdm
from utils import get_model, find_delimiter
import torch
import cProfile
import pstats


# Convertir texto a embedding
def enconde_text(model_name, model, text):

    if model_name == 'uae-large':
        return model.encode(text, to_numpy=True)
    else:
        return [model.encode(text, show_progress_bar=False)]


# Saca embeddings de cada fila
def content_embeddings(model, df, pool):#size, model_name):# pool
    # Array vacio donde meter todos y luego promediarlos
    #all_embs = np.empty((0, size), dtype=np.float32)

    sentences = [" ".join(map(str, row.values.flatten().tolist())) for _, row in df.iterrows()]

    all_embs = model.encode_multi_process(sentences, pool)
   
    #for _, row in df.iterrows():

    #    text = " ".join(map(str, row.values.flatten().tolist()))
        # Create embedding from chunks
    #    embs = enconde_text(model_name, model, text)

    #    all_embs = np.append(all_embs, embs, axis=0)

    return all_embs

@profile
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
    args.input = '../data/' + args.input + '/'
    files = os.listdir(args.input)

    torch.set_num_threads(4)

    models = []
    if args.model == 'all':
        models = ['uae-large', 'bge-large', 'bge-base', 'gte-large', 'ember']
    else:
        models.append(args.model)

    for m in models:
        model, dimensions = get_model(m)
        model.max_seq_length = 512
        pool = model.start_multi_process_pool()

        index = faiss.IndexIDMap(faiss.IndexFlatIP(dimensions))

        # Id del dataset en el indice
        id = 0
        # Dataframe que guarda los emparejamiento de id - archivo
        map = pd.DataFrame()
        discarted_files = 0

        for file in tqdm(files[:10]):

            try:
                # Read dataframe
                delimiter = find_delimiter(args.input + file)
                df = pd.read_csv(args.input + file, sep=delimiter)

                # Remove columns with all NaNs
                df = df.dropna(axis='columns', how='all')
                df.dropna(how='all', inplace=True)

                ##############
                #  BASELINE  #
                ##############

                if dataset != 'wikitables':

                    # Se mezcla
                    df = sklearn.utils.shuffle(df, random_state=0)

                    # Seleccionamos % de la tabla que sera indexado
                    ratio = 0.90
                    total_rows = df.shape[0]
                    index_size = int(total_rows*ratio)

                    # Datos en para indexar
                    df_index = df[0:index_size]

                    # Se calculan los embeddings
                    embs = content_embeddings(model, df_index, dimensions, m)
                    embs = np.array([np.mean(embs, axis=0)])
                    # Se normalizan
                    faiss.normalize_L2(embs)

                    # Se indexan y se las pasa un id
                    index.add_with_ids(embs, np.array([id]))
                    new_row = {"id": id, "dataset": file}
                    map = pd.concat([map, pd.DataFrame([new_row])],
                                    ignore_index=True)
                    id += 1

                else:
                    # Se calculan los embeddings
                    embs = content_embeddings(model, df, pool)#dimensions, m) #pool
                    embs = np.array([np.mean(embs, axis=0)])
                    # Se normalizan
                    faiss.normalize_L2(embs)

                    # Se indexan y se las pasa un id
                    index.add_with_ids(embs, np.array([id]))
                    new_row = {"id": id, "dataset": file}
                    map = pd.concat([map, pd.DataFrame([new_row])],
                                    ignore_index=True)
                    id += 1

            except Exception as e:
                print('Error en archivo', file)
                discarted_files += 1
                #print(e)
                #print(traceback.format_exc())

        model.stop_multi_process_pool(pool)
        print("Discarted_files:", discarted_files)

        faiss.write_index(index, "./index_files/"+m+"_"+dataset+".index")
       c


if __name__ == "__main__":
    main()
