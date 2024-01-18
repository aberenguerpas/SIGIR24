import argparse
import os
from extract_embeddings import extract_base_embeddings
from test_embeddings import test_random_reording_embeddings, test_random_deletion_of_columns, test_header_vector

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
    parser.add_argument('-g', '--gpu', default='gpu0')
    
    args = parser.parse_args()

    dataset = args.input
    args.input = '../data/' + args.input + '/'

    # Choose GPU
    if args.gpu == 'gpu0' and dataset == 'wikitables':
        args.input = '../data/' + args.input + '/' + args.gpu + '/'
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    elif args.gpu == 'gpu5' and dataset == 'wikitables':
        args.input = '../data/' + args.input + '/' + args.gpu + '/'
        os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

    files = os.listdir(args.input)

    models = []
    if args.model == 'all':
        models = ['uae-large', 'bge-large', 'bge-base', 'gte-large', 'ember']
    else:
        models.append(args.model)

    # 1. Extract base embeddings
    extract_base_embeddings(args, dataset, files, models)

    # 2. Random reordering of its columns test
    # test_random_reording_embeddings(args, dataset, files, models)

    # 3. Random deleting 50% of columns test
    # test_random_deletion_of_columns(args, dataset, files, models)

    # 5. Compare with header row test
    # test_header_vector(args, dataset, files, models)

if __name__ == "__main__":
    main()