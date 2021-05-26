import numpy as np

b = np.zeros((1,9000), dtype=np.float32)
a = np.array([1,2])

b[0][-a.shape[0]:] = a[-9000:]

