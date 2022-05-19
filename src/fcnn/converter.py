#converts the tt-format into the corresponding matrix

import numpy as np
import torch

d = 5
in_modes = np.array([2, 2, 2, 2, 2])
out_modes = np.array([3, 3, 3, 3, 3])
ranks = [1, 4, 5, 6, 7, 1]
cores = [torch.randn(in_modes[i], ranks[i], ranks[i+1], out_modes[i]) for i in range(d)]


def index5_to_1(n, d0, d1, d2, d3, d4):
	return np.prod(n[1:])*d0 + np.prod(n[2:])*d1 + np.prod(n[3:])*d2 + np.prod(n[4:])*d3 + d4 

def TT_to_matrix(cores, d, in_modes, out_modes, ranks):
	n = in_modes * out_modes
	w = np.zeros(np.prod(in_modes) * np.prod(out_modes))
	for i in range(d):
		cores[i] = torch.permute(cores[i], (0,3, 1,2))
		cores[i] = torch.reshape(cores[i], (-1, ranks[i], ranks[i+1]))

	for d0 in range(cores[0].size()[0]):
		for d1 in range(cores[1].size()[0]):
			for d2 in range(cores[2].size()[0]):
				for d3 in range(cores[3].size()[0]):
					for d4 in range(cores[4].size()[0]):
						w[index5_to_1(n, d0, d1, d2, d3, d4)] = cores[0][d0] @ cores[1][d1] @ cores[2][d2] @ cores[3][d3] @ cores[4][d4]
	w = w.reshape(np.prod(in_modes), np.prod(out_modes))
	w = np.transpose(w)
	return w

if __name__ == '__main__':  
	TT_to_matrix(cores, d, in_modes, out_modes, ranks)
