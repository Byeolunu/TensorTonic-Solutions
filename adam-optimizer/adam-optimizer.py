import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    #first moment calculation : 
    m_new=beta1*m+(1-beta1)*grad
    #second moment calculation : 
    v_new=beta2*v+(1-beta2)*grad**2

    #corrected 1st mmt  :
    m_=m_new/(1-beta1**t)
    #corrected 2nd mmt:
    v_=v_new/(1-beta2**t)

    #update
    param_new=param-lr*(m_/(np.sqrt(v_)+eps))

    return param_new,m_new,v_new
