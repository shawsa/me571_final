import sys, getopt

import numpy as np
import scipy.sparse as sp
from halton import halton_sequence
from scipy.spatial import cKDTree
from array import array

def halton(n):
    inner_nodes = halton_sequence(1,n,2).T
    inner_nodes = [(np.sqrt(x)*np.cos(2*np.pi*y), 
                             np.sqrt(x)*np.sin(2*np.pi*y)) 
                            for (x,y) in inner_nodes]
    return inner_nodes

def vogel(n):
    theta_hat = np.pi*(3-np.sqrt(5))
    inner_nodes = [ (np.sqrt(i/n)*np.cos(i*theta_hat), 
                              np.sqrt(i/n)*np.sin(i*theta_hat)) for i in range(1,n+1)]
    return inner_nodes


def boundary_param(t):
    return (np.cos(2*np.pi*t), np.sin(2*np.pi*t))


def gen_points(n, n_boundary, dist='vogel', boundary_dist='vogel', sorting='x'):
    if dist == 'vogel':
        inner_nodes = vogel(n)
    elif dist == 'halton':
        inner_nodes = halton(n)
    else:
        raise ValueError('dist=' + dist + ' not recognized')

    if boundary_dist=='equal':
        boundary_nodes = [
            (np.cos(t), np.sin(t))
            for t in 
            np.linspace(0, 2*np.pi, n_boundary, endpoint=False)]
    elif boundary_dist=='vogel':
        theta_hat = np.pi*(3-np.sqrt(5))
        boundary_nodes = [
            (np.cos(i*theta_hat), np.sin(i*theta_hat))
            for i in range(n+1, n+1 + n_boundary)]
    else:
        raise ValueError('boundary_dist=' + boundary_dist + ' not recognized')

    if sorting==None:
        pass
    elif sorting=='x':
        #sort by x value
        if type(inner_nodes)==np.ndarray:
            inner_nodes.sort(axis=0)
        else:
            inner_nodes.sort(key=lambda x: x[0])

        if type(boundary_nodes)==np.ndarray:
            boundary_nodes.sort(axis=0)
        else:
            boundary_nodes.sort(key=lambda x: x[0])
    else:
        raise ValueError('sorting=' + sorting + ' not recognized')

    return inner_nodes, boundary_nodes

def write_points_to_file(inner, boundary, stencil_size, filename=None):
    # generate nearest neighbors
    nodes = inner + boundary
    tree = cKDTree(np.array(nodes))
    nn = [tree.query(node, stencil_size)[1] for node in nodes]

    if filename==None:
        filename = ('n'+ str(len(inner)) + '_nb' + 
                    str(len(boundary)) + '_l' + 
                    str(stencil_size) +'.dat')
    

    f = open( 'point_sets/' + filename, 'wb')

    # write n, nb, l
    n = len(inner)
    f.write(n.to_bytes(4,'little'))
    nb = len(boundary)
    f.write(nb.to_bytes(4,'little'))
    f.write(stencil_size.to_bytes(4,'little'))
    
    # write two dummy ints for compatability with read mat
    fakeint = int(0)
    f.write(fakeint.to_bytes(4,'little'))
    f.write(fakeint.to_bytes(4,'little'))

    # write xs
    my_array = array('d', [node[0] for node in nodes])
    my_array.tofile(f)

    # write ys
    my_array = array('d', [node[1] for node in nodes])
    my_array.tofile(f)

    my_array = array('i', [v for row in nn for v in row])
    my_array.tofile(f)
    
    f.close()

def gen_points_file(
        n, n_boundary, stencil_size, dist='vogel', 
        boundary_dist='equal', sorting='x', 
        filename=None):
    
    inner, boundary = gen_points(
        n, n_boundary, dist=dist, 
        boundary_dist=boundary_dist, sorting=sorting)
    write_points_to_file(inner, boundary, stencil_size, filename)

def read_points(filename):
    f = open(filename, 'rb')
    n = int.from_bytes(f.read(4), 'little')
    nb = int.from_bytes(f.read(4), 'little')
    l_max = int.from_bytes(f.read(4), 'little')
    l = int.from_bytes(f.read(4), 'little')
    deg = int.from_bytes(f.read(4), 'little')
    
    xs = np.fromfile(f, dtype='d', count=n+nb)
    ys = np.fromfile(f, dtype='d', count=n+nb)
    nn = np.fromfile(f, dtype='i', count=n*l).reshape((n, l_max))
    
    #return xs, ys, nn
    nodes = [(x,y) for x,y in zip(xs,ys)]
    inner = nodes[:n]
    boundary = nodes[n:]
    return inner, boundary

def read_matrix(filename):
    f = open(filename, 'rb')
    n = int.from_bytes(f.read(4), 'little')
    nb = int.from_bytes(f.read(4), 'little')
    l_max = int.from_bytes(f.read(4), 'little')
    l = int.from_bytes(f.read(4), 'little')
    deg = int.from_bytes(f.read(4), 'little')
    pdim = (deg+1)*(deg+2)//2

    #print(n, nb, l_max, l, deg)
    
    xs = np.fromfile(f, dtype='d', count=n+nb)
    ys = np.fromfile(f, dtype='d', count=n+nb)
    nn = np.fromfile(f, dtype='i', count=n*l_max).reshape((n, l_max))
    weights = np.fromfile(f, dtype='d', count=n*(l+pdim)).reshape((n,l+pdim))

    row_index = [r for r in range(n) for c in range(l)]
    col_index = nn[:, :l].ravel()
    C = sp.csc_matrix((weights[:,:l].ravel(), (row_index, col_index)),shape=(n,n+nb))
    #print(len(row_index), len(col_index), len(weights))
    
    #return xs, ys, C
    nodes = [(x,y) for x,y in zip(xs,ys)]
    inner = nodes[:n]
    boundary = nodes[n:]
    return inner, boundary, C, l, pdim

if __name__ == '__main__':
    try:
      opts, args = getopt.getopt(sys.argv[1:], "n:b:l:")
    except getopt.GetoptError:
      print('GetOptError')
      sys.exit(2)
    for opt, arg in opts:
        if opt == '-n':
            n = int(arg)
        elif opt == "-b":
            nb = int(arg)
        elif opt == "-l":
            l = int(arg)
    filename = 'point_sets/n' + str(n) + '_' + 'nb' + str(nb) + '_' + 'l' + str(l) + '.dat'
    gen_points_file(n, nb, l, filename=filename)
    
            

