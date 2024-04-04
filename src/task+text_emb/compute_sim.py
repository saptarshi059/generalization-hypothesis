import argparse
import numpy as np
import torch
import os


def compute_sim(v1, v2):
    vector1 = torch.from_numpy(np.load(v1))
    vector2 = torch.from_numpy(np.load(v2))

    cos = torch.nn.CosineSimilarity(dim=0)
    print(f'Cosine Similarity between {args.dataset1} and {args.dataset2}: {cos(vector1, vector2).item()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset1', type=str)
    parser.add_argument('--dataset2', type=str)
    parser.add_argument('--embedding_type', default='text', choices=['text', 'task'])
    args = parser.parse_args()

    if args.embedding_type == 'text':
        embedding1_file = os.path.abspath(f'../../data/json_data/{args.dataset1}/avg_sequence_output.npy')
        embedding2_file = os.path.abspath(f'../../data/json_data/{args.dataset2}/avg_sequence_output.npy')
        compute_sim(embedding1_file, embedding2_file)

    else:
        for i in range(12):
            embedding1_file = os.path.abspath(f'../../data/json_data/{args.dataset1}/'
                                              f'task_emb/encoder.layer.{i}.layer_output.npy')
            embedding2_file = os.path.abspath(f'../../data/json_data/{args.dataset2}/task_emb/'
                                              f'encoder.layer.{i}.layer_output.npy')
            compute_sim(embedding1_file, embedding2_file)
