from sentence_transformers import SentenceTransformer
from angle_emb import AnglE
import csv


def get_model(name):

    if name == 'uae-large':
        model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
        dimensions = 1024

    if name == 'bge-large':
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        dimensions = 1024

    if name == 'bge-base':
        model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        dimensions = 768

    if name == 'gte-large':
        model = SentenceTransformer('thenlper/gte-large')
        dimensions = 1024

    if name == 'ember':
        model = SentenceTransformer('llmrails/ember-v1')
        dimensions = 1024

    return model, dimensions


def find_delimiter(filename):
    sniffer = csv.Sniffer()
    with open(filename) as fp:
        delimiter = sniffer.sniff(fp.read(5000)).delimiter
    return delimiter