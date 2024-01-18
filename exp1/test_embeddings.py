import faiss
import pandas as pd
import numpy as np
import csv
import os
import traceback
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_model, find_delimiter, content_embeddings, recover_data

def test_random_reording_embeddings(args, dataset, files, models):
    for m in models:
        model, tokenizer, dimensions = get_model(m)
        model.max_seq_length = 512
        avg_similarities = np.array(0)
        std_similarities = np.array(0)

        for file in tqdm(files):
            try:
                # Read dataframe
                delimiter = find_delimiter(args.input + file)
                df = pd.read_csv(args.input + file, sep=delimiter)

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

        directory = 'results/random_reordering/' + dataset + '/'
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # Write values in a CSV file
        file_name = directory + m + '_' + dataset + '.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            writer.writerow(values)

def test_random_deletion_of_columns(args, dataset, files, models):
    for m in models:
        model, tokenizer, dimensions = get_model(m)
        model.max_seq_length = dimensions
        avg_similarities = np.array(0)
        std_similarities = np.array(0)

        for file in tqdm(files):
            try:
                # Read dataframe
                delimiter = find_delimiter(args.input + file)
                df = pd.read_csv(args.input + file, sep=delimiter)

                # Remove columns with all NaNs
                df = df.dropna(axis='columns', how='all')
                df.dropna(how='all', inplace=True)

                # Random deletion of columns
                columns_to_delete = np.random.choice(df.columns, size=int(df.shape[1] * 0.5), replace=False)
                df = df.drop(columns=columns_to_delete)

                # Calculate embeddings
                embs = content_embeddings(model, df, dimensions, m, tokenizer)

                # Load original index of embeddings
                index = faiss.read_index('embeddings/' + dataset + '/' + m + '_' + dataset + '_' + file.replace('.csv', '') + '.index')

                # Recover original embeddings
                base_embs = recover_data(index)

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

        directory = 'results/random_deletion_of_columns/' + dataset + '/'
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # Write values in a CSV file
        file_name = directory + m + '_' + dataset + '.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            writer.writerow(values)

def test_header_vector(args, dataset, files, models):
    for m in models:
        model, tokenizer, dimensions = get_model(m)
        model.max_seq_length = dimensions
        avg_similarities = np.array(0)
        std_similarities = np.array(0)

        for file in tqdm(files):
            try:
                # Read dataframe
                delimiter = find_delimiter(args.input + file)
                df = pd.read_csv(args.input + file, sep=delimiter, nrows=1)

                # Calculate header embedding
                embs = content_embeddings(model, df, dimensions, m, tokenizer, header=True)

                # Load original index of embeddings
                index = faiss.read_index('embeddings/' + dataset + '/' + m + '_' + dataset + '_' + file.replace('.csv', '') + '.index')

                # Recover original embeddings
                base_embs = recover_data(index)

                # Compare original embbedings with embeddings obtained after having mixed the table
                similarity_scores = np.cosine_similarity(base_embs, embs)

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

        directory = 'results/header_vector/' + dataset + '/'
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # Write values in a CSV file
        file_name = directory + m + '_' + dataset + '.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            writer.writerow(values)