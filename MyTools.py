import numpy as np

def im2col(x,b):
    y = []
    for h in range(x.shape[0]/b):
        for v in range(x.shape[1]/b):
            y.append(x[h*b:h*b+b, v*b:v*b+b].flatten('F').tolist())
    return np.array(y).T

def col2im(x,b,xshape):
    y = np.zeros(xshape)
    for h in range(y.shape[0]/b):
        for v in range(y.shape[1]/b):
            j = h*(y.shape[0]/b) + v
            y[h*b:h*b+b, v*b:v*b+b] = x[:,j].reshape(b,b,order='F')
    return y

def dct2mtx(b):
    x = np.zeros((b,b))
    for k in np.arange(b):
        v = np.cos((np.arange(b)+.5) * k * np.pi / b)
        if k > 0:
            v = v - v.mean()
        x[:,k] = v / np.sqrt((v**2).sum())
    x = np.kron(x,x)
    return x