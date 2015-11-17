# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:23:09 2015

@author: Marius Felix Killinger
"""

import numpy as np
import scipy.sparse
from scipy.misc import comb
from scipy import sparse

from _malis import malis_loss_weights, connected_components, marker_watershed

def compute_V_rand_N2(seg_true, seg_pred):
    # TODO which RI is the right one???
    """
    Computes Rand index of ``seg_pred`` w.r.t ``seg_true``.
    The input arrays both contain label IDs and
    may be of arbitrary, but equal, shape.
    
    Pixels which are have ID in the true segmentation are not counted!
    
    Parameters:
    
    seg_true: np.ndarray
      True segmentation, IDs
    seg_pred: np.ndarray
      Predicted segmentation
    
    Returns
    -------
    
    ri: ???
    """
    seg_true = seg_true.ravel()
    seg_pred = seg_pred.ravel()
    idx = seg_true != 0
    seg_true = seg_true[idx]
    seg_pred = seg_pred[idx]

    cont_table = scipy.sparse.coo_matrix((np.ones(seg_true.shape),(seg_true,seg_pred))).toarray()
    P = cont_table/cont_table.sum()
    t = P.sum(axis=0)
    s = P.sum(axis=1)

    V_rand_split = (P**2).sum() / (t**2).sum()
    V_rand_merge = (P**2).sum() / (s**2).sum()
    V_rand = 2*(P**2).sum() / ((t**2).sum()+(s**2).sum())

    return (V_rand, V_rand_split, V_rand_merge)

#def rand_index(seg_true, seg_pred):
#    # TODO which RI is the right one???
#    """
#    Computes Rand index of ``seg_pred`` w.r.t ``seg_true``.
#    The input arrays both contain label IDs and
#    may be of arbitrary, but equal, shape.
#    
#    Pixels which are have ID in the true segmentation are not counted!
#    
#    Parameters:
#    
#    seg_true: np.ndarray
#      True segmentation, IDs
#    seg_pred: np.ndarray
#      Predicted segmentation
#    
#    Returns
#    -------
#    
#    ri: float
#      Rand Index
#    fscore: float
#      F score
#    prec: float
#      Precision
#    rec: float
#      Recall
#    """
#    seg_true = seg_true.ravel()
#    seg_pred = seg_pred.ravel()
#    idx = (seg_true != 0)
#    seg_true = seg_true[idx]
#    seg_pred = seg_pred[idx]
#
#    tp_plus_fp = comb(np.bincount(seg_true), 2).sum()
#    tp_plus_fn = comb(np.bincount(seg_pred), 2).sum()
#    A = np.c_[(seg_true, seg_pred)]
#    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
#             for i in set(seg_true))
#    fp = tp_plus_fp - tp
#    fn = tp_plus_fn - tp
#    tn = comb(len(A), 2) - tp - fp - fn
#    ri = (tp + tn) / (tp + fp + fn + tn)
#    prec = tp/(tp+fp)
#    rec = tp/(tp+fn)
#    fscore = 2*prec*rec/(prec+rec)
#    return ri, fscore, prec, rec

#def mknhood2d(radius=1):
#    """
#    Makes nhood structures for some most used dense graphs
#    """
#
#    ceilrad = np.ceil(radius)
#    x = np.arange(-ceilrad,ceilrad+1,1)
#    y = np.arange(-ceilrad,ceilrad+1,1)
#    [i,j] = np.meshgrid(y,x)
#
#    idxkeep = (i**2+j**2)<=radius**2
#    i=i[idxkeep].ravel(); j=j[idxkeep].ravel();
#    zeroIdx = np.ceil(len(i)/2).astype(np.int32);
#
#    nhood = np.vstack((i[:zeroIdx],j[:zeroIdx])).T.astype(np.int32)
#    return np.ascontiguousarray(np.flipud(nhood))
#
#def mknhood3d(radius=1):
#    """
#    Makes nhood structures for some most used dense graphs.
#    The neighborhood reference for the dense graph representation we use
#    nhood(1,:) is a 3 vector that describe the node that conn(:,:,:,1) connects to
#    so to use it: conn(23,12,42,3) is the edge between node [23 12 42] and [23 12 42]+nhood(3,:)
#    See? It's simple! nhood is just the offset vector that the edge corresponds to.
#    """
#
#    ceilrad = np.ceil(radius)
#    x = np.arange(-ceilrad,ceilrad+1,1)
#    y = np.arange(-ceilrad,ceilrad+1,1)
#    z = np.arange(-ceilrad,ceilrad+1,1)
#    [i,j,k] = np.meshgrid(z,y,z)
#
#    idxkeep = (i**2+j**2+k**2)<=radius**2
#    i=i[idxkeep].ravel(); j=j[idxkeep].ravel(); k=k[idxkeep].ravel();
#    zeroIdx = np.ceil(len(i)/2).astype(np.int32);
#
#    nhood = np.vstack((k[:zeroIdx],i[:zeroIdx],j[:zeroIdx])).T.astype(np.int32)
#    return np.ascontiguousarray(np.flipud(nhood))
#
#def mknhood3d_aniso(radiusxy=1,radiusxy_zminus1=1.8):
#    """
#    Makes nhood structures for some most used dense graphs.
#    """
#
#    nhoodxyz = mknhood3d(radiusxy)
#    nhoodxy_zminus1 = mknhood2d(radiusxy_zminus1)
#    
#    nhood = np.zeros((nhoodxyz.shape[0]+2*nhoodxy_zminus1.shape[0],3),dtype=np.int32)
#    nhood[:3,:3] = nhoodxyz
#    nhood[3:,0] = -1
#    nhood[3:,1:] = np.vstack((nhoodxy_zminus1,-nhoodxy_zminus1))
#
#    return np.ascontiguousarray(nhood)

def bmap_to_affgraph(bmap, nhood, return_min_idx=False):
    """
    Construct an affinity graph from a boundary map
    
    The spatial shape of the affinity graph is the same as of seg.
    This means that some edges are are undefined and therefore treated as disconnected.
    If the offsets in nhood are positive, the edges with largest spatial index are undefined.    
    
    Parameters
    ----------
    
    bmap: 3d np.ndarray, int
        Volume of boundaries
        0: object interior, 1: boundaries / ECS 
    nhood: 2d np.ndarray, int
        Neighbourhood pattern specifying the edges in the affinity graph
        Shape: (#edges, ndim)
        nhood[i] contains the displacement coordinates of edge i
        The number and order of edges is arbitrary
        
    Returns
    -------
    
    aff: 4d np.ndarray int32
        Affinity graph of shape (#edges, x, y, z)
        1: connected, 0: disconnected
        
    """
    if np.max(abs(nhood))>1:
        raise ValueError("Cannot construct affinity graph with edges longer\
        than 1 pixel from boundary map. Please construct affinity graph from\
        segmentation / IDs.")
      
    nhood = np.ascontiguousarray(nhood, np.int32)
    shape = bmap.shape
    n_edge = nhood.shape[0]
    aff = np.ones((n_edge,)+shape,dtype=np.int32)

    for e in range(n_edge):
        aff[e, \
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = np.maximum( \
                        bmap[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])], \
                        bmap[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                            max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] )

                            
    aff = 1 - aff # we have actually accumulated edges which are disconnected by a boundary

    
    return aff

def seg_to_affgraph(seg, nhood):
    """
    Construct an affinity graph from a segmentation (IDs) 
    
    Segments with ID 0 are regarded as disconnected
    The spatial shape of the affinity graph is the same as of seg.
    This means that some edges are are undefined and therefore treated as disconnected.
    If the offsets in nhood are positive, the edges with largest spatial index are undefined.
    
    Parameters
    ----------
    
    seg: 3d np.ndarray, int
        Volume of segmentation IDs
    nhood: 2d np.ndarray, int
        Neighbourhood pattern specifying the edges in the affinity graph
        Shape: (#edges, ndim)
        nhood[i] contains the displacement coordinates of edge i
        The number and order of edges is arbitrary
        
    Returns
    -------
    
    aff: 4d np.ndarray int32
        Affinity graph of shape (#edges, x, y, z)
        1: connected, 0: disconnected        
    """  
    nhood = np.ascontiguousarray(nhood, np.int32)
    shape = seg.shape
    n_edge = nhood.shape[0]
    aff = np.zeros((n_edge,)+shape,dtype=np.int32)

    for e in range(n_edge):
        aff[e, \
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = \
                        (seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] == \
                         seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                            max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] ) \
                        * ( seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] > 0 ) \
                        * ( seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                            max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] > 0 )

    return aff



def nodelist_like(shape, nhood):
    """
    Constructs the two node lists to represent edges of
    an affinity graph for a given volume shape and neighbourhood pattern.
    
    Parameters
    ----------
    
    shape: tuple/list
     shape of corresponding volume (z, y, x)
    nhood: 2d np.ndarray, int
        Neighbourhood pattern specifying the edges in the affinity graph
        Shape: (#edges, ndim)
        nhood[i] contains the displacement coordinates of edge i
        The number and order of edges is arbitrary
        
    Returns
    -------
    
    node1: 4d np.ndarray, int32
        Start nodes, reshaped as array
    
    node2: 4d np.ndarray, int32
        End nodes, reshaped as array
    """
    n_edge = nhood.shape[0]
    nodes = np.arange(np.prod(shape),dtype=np.int32).reshape(shape)
    node1 = np.tile(nodes, (n_edge,1,1,1))
    node2 = np.full(node1.shape, -1, dtype=np.int32)

    for e in range(n_edge):
        node2[e, \
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = \
                nodes[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                     max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                     max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])]

    return node1, node2


#def affgraph_to_edgelist(aff, nhood):
#    """
#    Constructs the two node lists and a list of edge weights to represent edges of
#    an affinity graph for a given volume shape and neighbourhood pattern.
#    
#    Parameters
#    ----------
#    
#    shape: tuple/list
#     shape of corresponding volume (z, y, x)
#    nhood: 2d np.ndarray, int
#        Neighbourhood pattern specifying the edges in the affinity graph
#        Shape: (#edges, ndim)
#        nhood[i] contains the displacement coordinates of edge i
#        The number and order of edges is arbitrary
#        
#    Returns
#    -------
#    
#    node1: 1d np.ndarray, int
#        Start nodes
#    
#    node2: 1d np.ndarray, int
#        End nodes
#        
#    aff: 1d np.ndarray
#        Edge weight between node1 and node2
#    """    
#    node1, node2 = nodelist_like(aff.shape[1:], nhood)
#    return node1.ravel(), node2.ravel(), aff.ravel()
#
#def connected_components_from_affgraph(aff, nhood):
#    node1, node2, edge = affgraph_to_edgelist(aff, nhood)
#    seg, seg_sizes = connected_components(int(np.prod(aff.shape[1:])),node1,node2,edge)
#    seg = seg.reshape(aff.shape[1:])
#    return seg, seg_sizes
    
    
class GetMalisWeights(object):
    def __init__(self):
        self.edgelist_cache = dict()
        
    def __call__(self, affinity_pred, affinity_gt, nhood, size_thresh=1):
        """
        Computes MALIS loss weights
        
        Roughly speaking the malis weights quantify the impact of an edge in
        the predicted affinity graph on the resulting segmentation.
        
        1. create edge lists / or lookup cached
        2. create GT IDs from affinity_gt (run CC)
        3. do pos/neg malis computation        
        """
        sh     = affinity_pred.shape
        vol_sh = sh[1:]
        key = (vol_sh, nhood.tobytes()) # cannot hash np.ndarray directly
        if self.edgelist_cache.has_key(key):
            node1, node2 = self.edgelist_cache[key]
        else:
            node1, node2 = nodelist_like(vol_sh, nhood)
            node1, node2 = node1.ravel(), node2.ravel()
            self.edgelist_cache[key] = (node1, node2)
        
        
        affinity_pred = np.ascontiguousarray(affinity_pred.ravel(), dtype=np.float32)
        affinity_gt   = np.ascontiguousarray(affinity_gt.ravel(), dtype=np.float32)
        nhood         = np.ascontiguousarray(nhood, np.int32)
        size_thresh   = int(size_thresh)
        
        
        # CC on GT
        seg, seg_sizes = connected_components(int(np.prod(vol_sh)), node1, node2, affinity_gt, size_thresh)
         
        # MALIS
        edge_weights_pos = np.minimum(affinity_pred, affinity_gt) 
        pos_counts = malis_loss_weights(seg, node1, node2, edge_weights_pos, 1)
        edge_weights_neg = np.maximum(affinity_pred, affinity_gt)
        neg_counts = malis_loss_weights(seg, node1, node2, edge_weights_neg, 0)
        
        pos_counts = pos_counts.reshape(sh)
        neg_counts = neg_counts.reshape(sh)
        seg        = seg.reshape(vol_sh)
        
        return pos_counts, neg_counts, seg
        
get_malis_weights = GetMalisWeights()     
     
     
if __name__=="__main__":
    pass
#    n = mknhood3d(2)
#    a = np.zeros((6,6,6), dtype=np.int)
#    c = 2
#    for x,y,z in n:
#        a[c+x, c+y, c+z] = 1
#        
#    seg1 = np.array([1,1,1,2,2,2,3,3,0,0])
#    seg2 = np.array([6,6,6,5,5,5,7,1,8,8])
#    
#    print rand_index(seg1, seg2)
#    print compute_V_rand_N2(seg1, seg2)
#    print 1-rand_index_ISBI(seg1, seg2)

    