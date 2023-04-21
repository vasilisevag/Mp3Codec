import numpy as np

def make_mp3_analysisfb(h: np.ndarray, M: int) -> np.ndarray:
	"""
	"""
	H = np.zeros([len(h), M], dtype=np.float32)
	for i in range(1, M + 1):
		n = np.arange(h.shape[0], dtype=np.int64)
		freq_i = (2 * i - 1) * np.pi / (2.0 * M)
		phas_i = -(2 * i - 1) * np.pi / 4.0
		tmp = np.cos(freq_i * n + phas_i)
		x = np.multiply(h, tmp)
		H[:, i - 1] = x
	return H

def make_mp3_synthesisfb(h: np.ndarray, M: int) -> np.ndarray:
	"""
	"""
	H = make_mp3_analysisfb(h, M)
	L = len(h)
	G = np.flip(H, axis=0)
	return G