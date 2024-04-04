import argparse
import numpy as np
import torch
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset1', type=str)
    parser.add_argument('--dataset2', type=str)
    parser.add_argument('--embedding_type', default='text', choices=['text', 'task'])
    args = parser.parse_args()

    if args.embedding_type == 'text':
        embedding1_file = os.path.abspath(f'../../data/json_data/{args.dataset1}/avg_sequence_output.npy')
        embedding2_file = os.path.abspath(f'../../data/json_data/{args.dataset2}/avg_sequence_output.npy')

        vector1 = torch.from_numpy(np.load(embedding1_file))
        vector2 = torch.from_numpy(np.load(embedding2_file))

        cos = torch.nn.CosineSimilarity(dim=0)
        print(f'Cosine Similarity between {args.dataset1} and {args.dataset2}: {cos(vector1, vector2).item()}')
    else:
        pass
