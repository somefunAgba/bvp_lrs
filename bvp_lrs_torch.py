import torch

def bvp_prof(r, *, m=0, e=0):
    '''Normalized BVP Profile: 

    Input Specs:
    - Required: 0 <= r <= 1
    - Optional: 0 <= m, e <= 1

    Use as `x = bvp_prof(r; m, e)`

    Output Specs:
    - 0 <= x <= 1
    
    CASE 1: explicit [2-point BVP] to [4-point BVP]
    - 0 <= m <= m+e <= 1

    CASE 2: explicit [3-point BVP] to [6-point BVP]
    - 0 <= m/2 <= m <= m+e <= (1+me)/2 <= 1
    '''
    me = m + e
    left, right = torch.inf, 0
    y = 1
    if m > 0:
        left = r/m
        y = torch.minimum(left, y)
    if me < 1:
        right = (1-r)/(1-me)
        y = torch.minimum(y, right)

    return 1 - y


def endpt(y, l=0):
    ''''Terminal value of the BVP

    Apply to enforce a terminal boundary condition `l` on the input `y, where
    - 0 <= l < 1
    , where
    - 0 <= y <= 1
    '''
    return l + (1-l)*y


def case1_lin(x, p:float=1):
    '''CASE 1: [2-point BVP]
    Dirichlet Energy Optimal Window

    Specs:
    - 0 <= x <= 1
    - Optional: p >= 1
    '''
    return (1-x)**(1/p)

def case1_rcos(x, p:float=1):
    '''CASE 1: [2-point BVP]
    Tikhonov-regularized Dirichlet Energy Optimal Window

    Specs:
    - 0 <= x <= 1
    - Optional: p >= 1      
    '''
    # [0.5*(1 + torch.cos(torch.pi * x))]**(1/p)
    return torch.cos(0.5*torch.pi*x)**(2/p)

def case2_lin(x, p:float=1):
    '''CASE 2: [3-point BVP]
    Dirichlet Energy Optimal Window

    Specs:
    - 0 <= x <= 1
    - Optional: p >= 1
    '''
    return (1 + x - 2*x*x)**(1/p)

def case2_rcos(x, p:float=1):
    '''CASE 2: [3-point BVP]
    Tikhonov-regularized Dirichlet Energy Optimal Window

    Specs:
    - 0 <= x <= 1
    - Optional: p >= 1      
    '''
    hcx = 0.5*torch.cos(torch.pi*x)
    return (1 + hcx - 2*hcx*hcx)**(1/p)

