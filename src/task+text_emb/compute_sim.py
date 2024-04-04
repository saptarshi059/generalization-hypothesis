import argparse
import numpy as np
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding1_file', type=str)
    parser.add_argument('--embedding2_file', type=str)
    args = parser.parse_args()

    vector1 = torch.from_numpy(np.load(args.embedding1_file))
    vector2 = torch.from_numpy(np.load(args.embedding2_file))

    cos = torch.nn.CosineSimilarity(dim=0)
    print(f'Cosine Similarity between {args.embedding1_file} and '
          f'{args.embedding2_file}: {cos(vector1, vector2).item()}')
