import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    
    w = np.zeros(X.shape[1])
    b=0.0
    for i in range(steps+1):
        z=np.dot(X,w)+b
        p=_sigmoid(z)
        error=p-y
        #∇w=X^t(p−y)/N
        w=w-lr*(np.dot(X.T,error))/len(y)
        #∇b=Σ(p−y)/N
        b=b-lr*(np.sum(error))/len(y)

        # w=w-lr*dw
        # b=b-lr*db
    return w,b


    