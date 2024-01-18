from sentence_transformers import SentenceTransformer
from angle_emb import AnglE
from transformers import AutoTokenizer
import csv

def get_model(name):

    if name == 'uae-large':
        model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
        tokenizer = AutoTokenizer.from_pretrained('WhereIsAI/UAE-Large-V1')
        dimensions = 1024

    if name == 'bge-large':
        model = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cuda')
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
        dimensions = 1024

    if name == 'bge-base':
        model = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cuda')
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
        dimensions = 768

    if name == 'gte-large':
        model = SentenceTransformer('thenlper/gte-large', device='cuda')
        tokenizer = AutoTokenizer.from_pretrained('thenlper/gte-large')
        dimensions = 1024

    if name == 'ember':
        model = SentenceTransformer('llmrails/ember-v1', device='cuda')
        tokenizer = AutoTokenizer.from_pretrained('llmrails/ember-v1')
        dimensions = 1024

    return model, tokenizer, dimensions


def find_delimiter(filename):
    sniffer = csv.Sniffer()
    with open(filename) as fp:
        delimiter = sniffer.sniff(fp.read(5000)).delimiter
    return delimiter