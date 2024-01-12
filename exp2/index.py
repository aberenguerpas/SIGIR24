import argparse
import faiss
import os
import pandas as pd
import numpy as np
import sklearn
import traceback
from tqdm import tqdm
from utils import get_model, find_delimiter
from sklearn.cluster import KMeans


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

        index = faiss.IndexIDMap(faiss.IndexFlatIP(dimensions))

        # Id del dataset en el indice
        id = 0
        # Dataframe que guarda los emparejamiento de id - archivo
        map = pd.DataFrame()

        for file in tqdm(files):

            try:
                # Read dataframe
                delimiter = find_delimiter(args.input + file)
                df = pd.read_csv(args.input + file, sep=delimiter, nrows=100)

                # Remove columns with all NaNs
                df = df.dropna(axis='columns', how='all')
                df.dropna(how='all', inplace=True)

                ##############
                #  BASELINE  #
                ##############

                if args.type == 'baseline':

                    # Se mezcla
                    df = sklearn.utils.shuffle(df)

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
                    map = pd.concat([map, pd.DataFrame([new_row])], ignore_index=True)
                    id += 1

                ##############
                #   RANDOM   #
                ##############

                elif args.type == 'random':

                    # Se mezcla
                    df = sklearn.utils.shuffle(df)

                    # Seleccionamos % de la tabla que sera indexado
                    ratio = 0.10
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
                    map = pd.concat([map, pd.DataFrame([new_row])], ignore_index=True)
                    id += 1

                ##############
                #   CLUSTER  #
                ##############
                elif args.type == 'cluster':

                    # Se mezcla
                    df = sklearn.utils.shuffle(df)

                    # Seleccionamos % de la tabla que sera indexado
                    ratio = 0.90
                    total_rows = df.shape[0]
                    index_size = int(total_rows*ratio)

                    # Datos en para indexar
                    df_index = df[0:index_size]

                    # Sacamos el numero de clusters
                    num_clusters = int(total_rows*0.1)
                    print(num_clusters)

                    all_embs = content_embeddings(model, df_index, dimensions, m)

                    clustering_model = KMeans(n_clusters=num_clusters)

                    clustering_model.fit(all_embs)
                    cluster_assignment = clustering_model.labels_

                    clustered_rows = [[] for i in range(num_clusters)]
                    for row_id, cluster_id in enumerate(cluster_assignment):
                        clustered_rows[cluster_id].append(all_embs[row_id])

                    centroids = []
                    for i, cluster in enumerate(clustered_rows):
                        centroid = np.mean(cluster, axis=0)
                        centroids.append(centroid)

                    avg_centroids = np.array([np.mean(centroids, axis=0)])

                    # Se normalizan
                    faiss.normalize_L2(avg_centroids)

                    # Se indexan y se las pasa un id
                    index.add_with_ids(avg_centroids, np.array([id]))
                    new_row = {"id": id, "dataset": file}
                    map = pd.concat([map, pd.DataFrame([new_row])], ignore_index=True)
                    id += 1


            except Exception as e:
                print('Error en archivo', file)
                print(e)
                print(traceback.format_exc())


        faiss.write_index(index, "./index_files/"+args.type+"_"+m+"_"+dataset+".index")
        map.to_csv("./index_files/"+args.type+"_"+m+"_"+dataset+"_map.csv", index=False)


if __name__ == "__main__":
    main()
