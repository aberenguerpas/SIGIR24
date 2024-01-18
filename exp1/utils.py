from sentence_transformers import SentenceTransformer
from angle_emb import AnglE
from transformers import AutoTokenizer
import csv
import numpy as np
import torch

def get_model(name):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if name == 'uae-large':
        model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
        tokenizer = AutoTokenizer.from_pretrained('WhereIsAI/UAE-Large-V1')
        dimensions = 1024

    if name == 'bge-large':
        model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
        dimensions = 1024

    if name == 'bge-base':
        model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
        dimensions = 768

    if name == 'gte-large':
        model = SentenceTransformer('thenlper/gte-large', device=device)
        tokenizer = AutoTokenizer.from_pretrained('thenlper/gte-large')
        dimensions = 1024

    if name == 'ember':
        model = SentenceTransformer('llmrails/ember-v1', device=device)
        tokenizer = AutoTokenizer.from_pretrained('llmrails/ember-v1')
        dimensions = 1024

    return model, tokenizer, dimensions


def find_delimiter(filename):
    sniffer = csv.Sniffer()
    with open(filename) as fp:
        delimiter = sniffer.sniff(fp.read(5000)).delimiter
    return delimiter


# Convert text to embedding
def enconde_text(model_name, model, text):
    if model_name == 'uae-large':
        return model.encode(text, to_numpy=True)
    else:
        return [model.encode(text, show_progress_bar=False)]
    
# Extract embeddings from each row
def content_embeddings(model, df, size, model_name, tokenizer, pool, header=False):
    """
    all_embs = np.empty((0, size), dtype=np.float32)

    for _, row in df.iterrows():
        if (header == False):
            text = " ".join(map(str, row.values.flatten().tolist()))
        else:
            text = " ".join(map(str, row.index.tolist()))

        batch_dict = tokenizer(text,  max_length=512, return_attention_mask=False, padding=False, truncation=True)
        # Filter that the row has no more than 512 tokens
        if len(batch_dict['input_ids']) < 512:
            # Create embedding from chunks
            embs = enconde_text(model_name, model, text)
            all_embs = np.append(all_embs, embs, axis=0)
    """
    if (header == False):
        sentences = [" ".join(map(str, row.values.flatten().tolist())) for _, row in df.iterrows()]
    else:
        sentences = [" ".join(map(str, row.index.tolist())) for _, row in df.iterrows()]

    all_embs = model.encode_multi_process(sentences, pool)
   
    return all_embs


def recover_data(index):
    ids = np.arange(index.ntotal).astype('int64')
    base_embs = []

    for id in ids:
        base_embs.append(index.reconstruct_n(int(id), 1)[0])

    return np.array(base_embs)
