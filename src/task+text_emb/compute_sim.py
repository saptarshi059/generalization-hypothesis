import argparse
import numpy as np
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding1_file', type=str)
    parser.add_argument('--embedding2_file', type=str)
    args = parser.parse_args()

    vector1 = torch.from_numpy(np.load(args.embedding1_file).reshape(1, -1))
    vector2 = torch.from_numpy(np.load(args.embedding2_file).reshape(1, -1))

    cos = torch.nn.CosineSimilarity()
    print(f'Cosine Similarity between {args.embedding1_file} and '
          f'{args.embedding1_file}: {np.round(cos(vector1, vector2).item(),2)}')
