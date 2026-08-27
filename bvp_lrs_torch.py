import torch

# 2-point BVP
def x2(r): 
    '''[2-BVP]
    Specs:
    - 0 <= r <= 1
    '''
    return r

# 3-point BVP
def x3(r, m=0):
    '''[3-BVP]
    Specs:
    - 0 <= r <= 1
    - 0 <= m < 1
    '''
    # init.
    x = 0
    #
    if m != 0: x = torch.maximum( (m-r)/m, x)
    return torch.maximum(x, (r - m)/(1 - m))

# 4-point BVP
def x4(r, m=0, e=0):
    '''[4-BVP]
    Specs:
    - 0 <= r <= 1
    - 0 <= m <= m + e < 1
    '''
    me = m + e
    # init.
    x = 0
    #
    if m != 0: x = torch.maximum( (m-r)/m, x)
    if me < 1: x = torch.maximum((r - me)/(1 - me), x)
    return x


# 8-point BVPs
def x8M(r, m1, m2, m3, e1, e2, e3, c):
    '''M shape [8-BVP]
    Specs:
    - 0 <= r <= 1
    - 0 <= m1 <= m1 + e1 <= m2 <= m2 + e2 <= m3 <= m3 + e3 < 1
    '''
    m1e = m1 + e1
    m2e = m2 + e2
    m3e = m3 + e3
    ma = m2 - m1e
    mb = m3 - m2e
    h = 1-c

    # init.
    x = 0
    #
    if m1 != 0: x = torch.maximum( (m1-r)/m1, x)
    #
    if ma <= 0 and mb > 0:
        x = torch.maximum( 
            h * torch.minimum( (m3 -r)/mb, 1),  
            x )
    elif ma > 0 and mb <= 0:
        x = torch.maximum(
                h * torch.minimum( 1, 
                    (r - m1e)/ma 
                ), x )           
    elif ma > 0 and mb > 0:
        x = torch.maximum(
                h * torch.minimum( 
                    torch.minimum( (m3 -r)/mb, 1), 
                    (r - m1e)/ma
                ), x )   
    #      
    if m3e < 1:
        x = torch.maximum((r - m3e)/(1 - m3e), x)
    return x

def x8Dome(r, m1, m2, m3, e1, e2, e3, c):
    '''Dome shape [8-BVP]
    Specs:
    - 0 <= r <= 1
    - 0 <= m1 <= m1 + e1 <= m2 <= m2 + e2 <= m3 <= m3 + e3 < 1    
    '''
    m1e = m1 + e1
    m2e = m2 + e2
    m3e = m3 + e3
    ma = m2 - m1e
    mb = m3 - m2e
    h, cr = 1-c, c*r

    # init.
    x = 0
    #
    if m1 != 0: x = torch.maximum( (m1-cr)/m1, x)
    #
    if ma <= 0 and mb > 0:
        x = torch.maximum( h * torch.minimum((r - m2e)/mb, 1), x)
    if ma > 0 and mb <= 0:
        x = torch.maximum( h * torch.minimum( (m2-r)/ma, 1), x)
    elif ma > 0 and mb > 0:
        x = torch.maximum( 
                h * torch.minimum(
                        torch.maximum((r - m2e)/mb, 
                            (m2-r)/ma ), 
                    1), x)
    #
    if m3e < 1:
        last = (h + cr - m3e)/(1 - m3e)
        x = torch.where(r >= m3e, torch.maximum(last, x), x)
    return x


# Dirichlet Energy Optimal Window
def glin(x, *, p=1):
    '''
    p-th root of the linear window
    '''
    return (1-x)**(1/p)

# Regularized Dirichlet Energy Optimal Window
def grcos(x, *, p=1):
    '''
    p-th root of the raised cosine window
    '''
    # y = [0.5*(1 + torch.cos(torch.pi * x))]**(1/p)
    return torch.cos(0.5*torch.pi*x)**(2/p)
