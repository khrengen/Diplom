# TT-Liner layer for PyTorch

import numpy as np
import torch
import torch.nn as nn


class tt_LinearV2(nn.Module):

	# initialization:
	## input:
	###  in_modes - numpy array of input modes
	###  out_modes - numpy array of output modes
	###  ranks - numpy array of tt-ranks, first and last element must be 1
	###  bias - True if necessary to add a vector of bias(deafult True)

	def __init__(self, in_modes, out_modes, ranks, bias=True):
		super().__init__()
		self.bias = bias
		self.in_modes = in_modes
		self.out_modes = out_modes
		self.ranks = ranks
		self.d = in_modes.size
		self.biases = torch.nn.Parameter(torch.randn(np.prod(out_modes)))
		self.cores = nn.ParameterList([torch.nn.Parameter(torch.randn(self.in_modes[i], self.ranks[i],
										self.ranks[i+1], self.out_modes[i])) for i in range(self.d)])
		self.reset_parameters()

	# initialize parametrs of tt-cores by kaiming uniform initialization

	def reset_parameters(self):
		if self.d != 5:
			print('length of the in_modes must be 5, not {}'.format(self.d))
			exit(1)
		for i in range(self.d):
			torch.nn.init.kaiming_uniform_(self.cores[i], a=np.sqrt(5))
		fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.cores[0])
		bound = 1 / np.sqrt(fan_in)
		torch.nn.init.uniform_(self.biases, -bound, bound)

	#forward pass
	## vector-by-matrix product in tt-format implemenr=ted by einsum operation

	def forward(self, input):
		input_rsh = torch.reshape(input, tuple(np.append([-1], self.in_modes)))

		# n index of sample in the batch,
		# a, b, c, d, e - indices corresponding to input modes
		# h, i, j, k, l, m - indices corresponding to tt ranks
		# v, w, x, y, z - indices corresponding to output modes

		out = torch.einsum("nabcde,ahiv,bijw,cjkx,dkly,elmz", input_rsh, 
			self.cores[0], self.cores[1], self.cores[2], self.cores[3], self.cores[4])
		out = torch.reshape(out, (-1, np.prod(self.out_modes)))
		if self.bias:
			out = torch.add(out, self.biases)
		return out