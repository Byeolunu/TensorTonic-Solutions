import numpy as np

def rnn_step_backward(dh, cache):
    """
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h (shape: H,)
        dW: gradient wrt W               (shape: H x D)
        dU: gradient wrt U               (shape: H x H)
        db: gradient wrt bias            (shape: H,)
    """
    x_t,h_prev, W, U, h_t = cache

    #tanh backward
    dz = dh*(1-h_t**2)

    #gradient wrt input x_t
    dx_t= W.T @ dz

    # gradient wrt previous h
    dh_prev = U.T @ dz

    #gradient wrt U 
    dU = np.outer(dz,h_prev)

    #gradient wrt W
    dW=np.outer(dz,x_t)

    #gradient wrt bias
    db=dz

    return dx_t,dh_prev,dW,dU,db


