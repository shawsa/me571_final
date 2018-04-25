import numpy as np
import scipy.linalg as la

# sparse matrices
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from scipy.sparse.linalg import bicgstab, bicg, spilu, LinearOperator, gmres
from scipy.sparse.csgraph import reverse_cuthill_mckee as rcm

# point generation
from gen_points import *

#**************************************************************************
#
# Polynomial Basis Terms
#
#**************************************************************************
def poly_basis(k):
    ret = []
    deg = 0
    while len(ret)<k:
        for i in range(deg+1):
            ret += [lambda x, i=i, deg=deg, : x[0]**(deg-i) * x[1]**i]
        deg += 1
    return ret[:k]

def poly_basis_L(k):
    ret = []
    deg = 0
    while len(ret)<k:
        for i in range(deg+1):
            ret += [lambda x, i=i, deg=deg, : 
                      (deg-i)*(deg-i-1)*x[0]**(deg-i-2) * x[1]**i 
                    + x[0]**(deg-i) * i*(i-1)*x[1]**(i-2)
                   ]
        deg += 1
    #testing Center evauluation of Lp(x,y) at x,y. Simplifies to 1 and 0s.
    ret = [lambda x: 1]
    while len(ret)<k:
        ret += [lambda x: 0]
    #end testing
    return ret[:k]

#**************************************************************************
#
# RBFs
#
#**************************************************************************

def dist(x, y):
    return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

def rbf0(r):
    return r**3
def rbf0d(r):
    return 6*r

def rbf1(r):
    return r**7
def rbf1d(r):
    return 42*r**7


#**************************************************************************
#
# boundary
#
#**************************************************************************
def boundary(x):
    return x[0]

def gen_system(inner_nodes, boundary_nodes, l, pdim, rbf_tag='r^3'):
    if rbf_tag=='r^3':
        rbf = lambda x,y: rbf0(dist(x,y))
        rbfd = lambda x,y: rbf0d(dist(x,y))
    elif rbf_tag=='r^7':
        rbf = lambda x,y: rbf1(dist(x,y))
        rbfd = lambda x,y: rbf1d(dist(x,y))
    else:
        raise ValueError('rbf_tag=' + rbf_tag + ' not recognized')
    
    n = len(inner_nodes)
    n_b = len(boundary_nodes)

    nodes = np.concatenate((inner_nodes, boundary_nodes), axis=0)
    tree = cKDTree(nodes)

    pbasis = poly_basis(pdim)
    pbasisL = poly_basis_L(pdim)
    weights = np.zeros((n, l))
    #row_index = []
    row_index = [r for r in range(n) for c in range(l)]
    col_index = np.zeros((n, l))

    for r in range(n):
        n_index = tree.query(nodes[r], l)[1]
        neighbors = [nodes[i] for i in n_index]
        A = np.array([[rbf(x,y) for x in neighbors] for y in neighbors]).reshape((l,l))
        P = np.array([[p(x) for p in pbasis] for x in neighbors])
        AP = np.block([[A.reshape((l,l)), P],[P.T, np.zeros((pdim,pdim))]])
    
        rhs = np.array([rbfd(nodes[r], nodes[i]) for i in n_index])
        rhsp = np.array([pd(nodes[r]) for pd in pbasisL])
        rhs = np.block([rhs.ravel(), rhsp.ravel()])

        weights[r] = (la.pinv(AP)@rhs)[:l]
        col_index[r] = n_index
        
    C = sp.csr_matrix((weights.ravel(), (row_index, col_index.ravel())),shape=(n,n+n_b))

    b_vec = np.array([-boundary(x) for x in boundary_nodes])
    return C, b_vec


def rbf_fd(inner_nodes, boundary_nodes, l, pdim, rbf_tag='r^3'):
    n = len(inner_nodes)
    n_b = len(boundary_nodes)
    C, b_vec = gen_system(inner_nodes, boundary_nodes, l, pdim, rbf_tag=rbf_tag)
    u = spsolve(C[:,:n], C[:,n:]@b_vec)
    u = np.concatenate((u.ravel(), -b_vec.ravel()), axis=0)
    return u, C, b_vec



#**************************************************************************
#
# debugging
#
#**************************************************************************

def get_first_system(inner_nodes, boundary_nodes, l, pdim, rbf_tag='r^3'):
    if rbf_tag=='r^3':
        rbf = lambda x,y: rbf0(dist(x,y))
        rbfd = lambda x,y: rbf0d(dist(x,y))
    elif rbf_tag=='r^7':
        rbf = lambda x,y: rbf1(dist(x,y))
        rbfd = lambda x,y: rbf1d(dist(x,y))
    else:
        raise ValueError('rbf_tag=' + rbf_tag + ' not recognized')
    
    n = len(inner_nodes)
    n_b = len(boundary_nodes)

    nodes = np.concatenate((inner_nodes, boundary_nodes), axis=0)
    tree = cKDTree(nodes)

    pbasis = poly_basis(pdim)
    pbasisL = poly_basis_L(pdim)
    weights = np.zeros((n, l))
    #row_index = []
    row_index = [r for r in range(n) for c in range(l)]
    col_index = np.zeros((n, l))

# remove
    grab = True
    for r in range(n):
        n_index = tree.query(nodes[r], l)[1]
        neighbors = [nodes[i] for i in n_index]
        A = np.array([[rbf(x,y) for x in neighbors] for y in neighbors]).reshape((l,l))
        P = np.array([[p(x) for p in pbasis] for x in neighbors])
        AP = np.block([[A.reshape((l,l)), P],[P.T, np.zeros((pdim,pdim))]])

            
        rhs = np.array([rbfd(nodes[r], nodes[i]) for i in n_index])
        rhsp = np.array([pd(nodes[r]) for pd in pbasisL])
        rhs = np.block([rhs.ravel(), rhsp.ravel()])

# remove
        if grab:
            grab = False
            return AP, rhs        

        weights[r] = (la.pinv(AP)@rhs)[:l]
        col_index[r] = n_index
        
    C = sp.csr_matrix((weights.ravel(), (row_index, col_index.ravel())),shape=(n,n+n_b))

    b_vec = np.array([-boundary(x) for x in boundary_nodes])
    return C, b_vec








