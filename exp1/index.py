import argparse
import os
import time
from extract_embeddings import extract_base_embeddings, extract_random_reording_embeddings

def random_reordering():
    return 0

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

    # 1. Extract base embeddings
    # inicio = time.time()
    # extract_base_embeddings(args, dataset, files, models)
    # fin = time.time()

    # tiempo_transcurrido = fin - inicio

    # print("Tiempo transcurrido:", tiempo_transcurrido, "segundos")

    # 2. Random reordering of its columns test
    extract_random_reording_embeddings(args, dataset, files, models)

if __name__ == "__main__":
    main()