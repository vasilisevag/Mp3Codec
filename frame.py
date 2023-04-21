import numpy as np

def frame_sub_analysis(xbuff: np.ndarray, H: np.ndarray, q: int) -> np.ndarray:
	L, M = H.shape
	ind = np.zeros([q, L]) # N X 512
	ind[0, :] = np.arange(L) # 1 2 3 ... 512
	for i in range(1, q):
		ind[i, :] += ind[i - 1, :] + M
	ind = ind.astype(np.int64)

	X = xbuff[ind]  # 36 x 512 --- 512 x 32
	Y = np.einsum('ik,kj->ij', X, H)
	return Y


def frame_sub_synthesis(ybuff: np.ndarray, G: np.ndarray) -> np.ndarray:
	"""
	"""
	L, M = G.shape
	N = int(np.ceil(L / M))

	Gr = G.reshape(M, M * N, order='F').copy()
	Z = np.zeros([1152])

	for n in range(ybuff.shape[0] - N):
		tmp = ybuff[n:n + N, :].T.flatten()
		yr = np.expand_dims(tmp, axis=-1)
		z = np.dot(Gr, yr)
		Z[n * M:(n + 1) * M] = M * np.flip(z[:, 0])
	return Z.T.flatten()