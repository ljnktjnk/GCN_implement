from torch_geometric.utils.convert import from_networkx
import torch_geometric
from bpemb import BPEmb
from sentence_transformers import SentenceTransformer
from graph import Grapher

import torch
import os 
import random
import numpy as np
import glob

bpemb_en = BPEmb(lang="en", dim=100)
sent_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

def make_sent_bert_features(text):
    emb = sent_model.encode([text])[0]
    return emb

def make_graph(file, classes):
    """
    returns one big graph with unconnected graphs with the following:
    - x (Tensor, optional) – Node feature matrix with shape [num_nodes, num_node_features]. (default: None)
    - y (Tensor, optional) – Graph or node targets with arbitrary shape. (default: None)
    """
    connect = Grapher(file)
    G,_,_ = connect.graph_formation(export_graph=False)
    df = connect.relative_distance(export_document_graph=False) 
    individual_data = from_networkx(G)

    feature_cols = ['rd_b', 'rd_r', 'rd_t', 'rd_l','line_number', \
            'n_upper', 'n_alpha', 'n_numeric']

    text_features = np.array(df["object"].map(make_sent_bert_features).tolist()).astype(np.float32)
    numeric_features = df[feature_cols].values.astype(np.float32)

    features = np.concatenate((numeric_features, text_features), axis=1)
    features = torch.tensor(features)

    for col in df.columns:
        try:
            df[col] = df[col].str.strip()
        except AttributeError as e:
            pass

    df['label'] = df['label'].fillna('undefined')
    for i, cl in enumerate(classes):
        df.loc[df['label'] == cl, 'num_labels'] = i+1
        
    df.loc[df['label'] == 'undefined', 'num_labels'] = i+1

    assert df['num_labels'].isnull().values.any() == False, f'labeling error! Invalid label(s) present in {file}'
    labels = torch.tensor(df['num_labels'].values.astype(np.int))
    text = df['object'].values

    individual_data.x = features
    individual_data.y = labels
    individual_data.text = text
    
    return individual_data, df

def get_data(folder, save_fd, classes):
    files = glob.glob(f"{folder}/*.csv")

    train_list_of_graphs, test_list_of_graphs = [], []

    random.shuffle(files)

    training, testing = files[:int(len(files)*0.85)], files[int(len(files)*0.85):]

    for file in files:
        print(file)
        individual_data, _ = make_graph(file, classes)
        if file in training:
            train_list_of_graphs.append(individual_data)
        elif file in testing:
            test_list_of_graphs.append(individual_data)

    train_data = torch_geometric.data.Batch.from_data_list(train_list_of_graphs)
    train_data.edge_attr = None
    test_data = torch_geometric.data.Batch.from_data_list(test_list_of_graphs)
    test_data.edge_attr = None
    
    torch.save(train_data, os.path.join(save_fd, 'train_data.dataset'))
    torch.save(test_data, os.path.join(save_fd, 'test_data.dataset'))

if __name__ == "__main__":
    import argparse

    # set up parameter
    parser = argparse.ArgumentParser(description='Make graph data')
    parser.add_argument('-input', type=str, default="./GCN_data/csv", help='input folder csv')
    parser.add_argument('-output', type=str, default="./GCN_data/processed", help='output path to save data')
    parser.add_argument('-cls', nargs="+", default=["com_name", "com_pos", "time", "other"], help='array class')

    args = parser.parse_args().__dict__

    input_path = args['input']
    output_path = args['output']
    classes = args['cls']

    get_data(input_path, output_path, classes)
