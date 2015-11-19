# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:07:37 2015

@author: Marius Felix Killinger
"""
import numpy as np
import theano
from theano import gof
import theano.tensor as T
from theano.gradient import disconnected_type

import malis_utils

__all__ = ['malis_weights']

class MalisWeights(theano.Op):
    """
    Computes MALIS loss weights
    
    Roughly speaking the malis weights quantify the impact of an edge in
    the predicted affinity graph on the resulting segmentation.
      
    Parameters
    ----------
    affinity_pred: 4d np.ndarray float32
        Affinity graph of shape (#edges, x, y, z)
        1: connected, 0: disconnected        
    affinity_gt: 4d np.ndarray int16
        Affinity graph of shape (#edges, x, y, z)
        1: connected, 0: disconnected 
    seg_gt: 3d np.ndarray, int (any precision)
        Volume of segmentation IDs        
    nhood: 2d np.ndarray, int
        Neighbourhood pattern specifying the edges in the affinity graph
        Shape: (#edges, ndim)
        nhood[i] contains the displacement coordinates of edge i
        The number and order of edges is arbitrary
    unrestrict_neg: Bool
        Use this to relax the restriction on neg_counts. The restriction
        modifies the edge weights for before calculating the negative counts
        as: ``edge_weights_neg = np.maximum(affinity_pred, affinity_gt)``
        If unrestricted the predictions are used directly.
        
    Returns
    -------
      
    pos_counts: 4d np.ndarray int32
      Impact counts for edges that should be 1 (connect)      
    neg_counts: 4d np.ndarray int32
      Impact counts for edges that should be 0 (disconnect)  
      
      
    Outline
    -------
      
    - Computes for all pixel-pairs the MaxiMin-Affinity
    - Separately for pixel-pairs that should/should not be connected
    - Every time an affinity prediction is a MaxiMin-Affinity its weight is
      incremented by one in the output matrix (in different slices depending
      on whether that that pair should/should not be connected)
    """
    __props__ = ()

    def make_node(self, *inputs):
        inputs = list(inputs)
        if len(inputs)!=5:
            raise ValueError("MalisOp takes 5 inputs: \
          affinity_pred, affinity_gt, seg_gt, nhood, unrestrict_neg")
        
        inputs = map(T.as_tensor_variable, inputs)
        
        affinity_pred, affinity_gt, seg_gt, nhood = inputs[:4]
        if affinity_pred.ndim!=4:
            raise ValueError("affinity_pred must be convertible to a\
          TensorVariable of dimensionality 4. This one has ndim=%i" \
                             %affinity_pred.ndim)

        if affinity_gt.ndim!=4:
            raise ValueError("affinity_gt must be convertible to a\
          TensorVariable of dimensionality 4. This one has ndim=%i" \
                             %affinity_gt.ndim)

        if seg_gt.ndim!=3:
            raise ValueError("seg_gt must be convertible to a\
          TensorVariable of dimensionality 3. This one has ndim=%i" \
                             %seg_gt.ndim)

        if nhood.ndim!=2:
            raise ValueError("nhood must be convertible to a\
          TensorVariable of dimensionality 2. This one has ndim=%i" \
                             %nhood.ndim)
                                     
        malis_weights_pos = T.TensorType(
            dtype='int32',
            broadcastable=(False,)*4)()

        malis_weights_neg = T.TensorType(
            dtype='int32',
            broadcastable=(False,)*4)()

        outputs = [malis_weights_pos, malis_weights_neg]

        return gof.Apply(self, inputs, outputs)

    def perform(self, node, inputs, output_storage):
        affinity_pred, affinity_gt, seg_gt, nhood, unrestrict_neg = inputs
        pos, neg = output_storage
        pos[0], neg[0] = malis_utils.malis_weights(affinity_pred,
                                                   affinity_gt,
                                                   seg_gt,
                                                   nhood,
                                                   unrestrict_neg)

    def grad(self, inputs, outputs_gradients):
        # The gradient of all outputs is 0 w.r.t to all inputs
        return [disconnected_type(),]*5

    def connection_pattern(self, node):
        # The gradient of all outputs is 0 w.r.t to all inputs
        return [[False, False],]*5

def malis_weights(affinity_pred, affinity_gt, seg_gt, nhood, unrestrict_neg=False):
    """
    Computes MALIS loss weights
    
    Roughly speaking the malis weights quantify the impact of an edge in
    the predicted affinity graph on the resulting segmentation.
      
    Parameters
    ----------
    affinity_pred: 4d np.ndarray float32
        Affinity graph of shape (#edges, x, y, z)
        1: connected, 0: disconnected        
    affinity_gt: 4d np.ndarray int16
        Affinity graph of shape (#edges, x, y, z)
        1: connected, 0: disconnected 
    seg_gt: 3d np.ndarray, int (any precision)
        Volume of segmentation IDs        
    nhood: 2d np.ndarray, int
        Neighbourhood pattern specifying the edges in the affinity graph
        Shape: (#edges, ndim)
        nhood[i] contains the displacement coordinates of edge i
        The number and order of edges is arbitrary
    unrestrict_neg: Bool
        Use this to relax the restriction on neg_counts. The restriction
        modifies the edge weights for before calculating the negative counts
        as: ``edge_weights_neg = np.maximum(affinity_pred, affinity_gt)``
        If unrestricted the predictions are used directly.
        
    Returns
    -------
      
    pos_counts: 4d np.ndarray int32
      Impact counts for edges that should be 1 (connect)      
    neg_counts: 4d np.ndarray int32
      Impact counts for edges that should be 0 (disconnect)  
      
      
    Outline
    -------
      
    - Computes for all pixel-pairs the MaxiMin-Affinity
    - Separately for pixel-pairs that should/should not be connected
    - Every time an affinity prediction is a MaxiMin-Affinity its weight is
      incremented by one in the output matrix (in different slices depending
      on whether that that pair should/should not be connected)
    """
    rest = 1 if unrestrict_neg else 0 # Theano cannot bool        
    return MalisWeights()(affinity_pred, affinity_gt, seg_gt, nhood, rest)    



if __name__=="__main__":
    import theano.sandbox.cuda
    theano.sandbox.cuda.use("gpu0")
    import theano
    
    print "TESTING/DEMO:"
    
    aff_pred_t = T.TensorType('float32', [False,]*4, name='aff_pred')()
    aff_gt_t   = T.TensorType('int16',   [False,]*4, name='aff_gt')()
    seg_gt_t   = T.TensorType('int16',   [False,]*3, name='seg_gt')()
    neigh_t    = T.TensorType('int32',   [False,]*2, name='neighb')()
    
    
    pos_t, neg_t = malis_weights(aff_pred_t, aff_gt_t, seg_gt_t, neigh_t)
    loss_t = T.sum(pos_t * aff_pred_t)
    
    f = theano.function([aff_pred_t, aff_gt_t, seg_gt_t, neigh_t], [pos_t, neg_t, loss_t])
    grad_t = theano.grad(loss_t, aff_pred_t)
    f2 = theano.function([aff_pred_t, aff_gt_t, seg_gt_t, neigh_t], [grad_t,])
    
    nhood = np.array([[ 0.,  1.,  0.],
                      [ 0.,  0.,  1.]], dtype=np.int32)

    test_id2 = np.array([[[1, 1, 2, 2, 0, 3],
                          [1, 1, 2, 2, 0, 3],
                          [1, 1, 2, 2, 0, 3],
                          [1, 1, 2, 2, 0, 3]]], dtype=np.int32)

    aff_gt   = malis_utils.seg_to_affgraph(test_id2, nhood)
    seg_gt   = malis_utils.affgraph_to_seg(aff_gt, nhood)[0].astype(np.int16)
    aff_pred = np.array([ [[[ 1.,  1.,  1.,  1.,  0., 1.],
                            [ 1.,  1.,  1.,  1.,   0., 1.],
                            [ 0.9, 0.8, 1.,  1.,  0., 1.],
                            [ 0.,  0.,  0.,  0.,  0., 1.]]],

                          [[[ 1.,  0.,  1.,  0.3, 0.4,0.],
                            [ 0.7, 0.,  1.,  0.,  0., 0.],
                            [ 1.,  0.2, 1.,  0.,  0., 0.],
                            [ 1.,  0.,  1.,  0.,  0., 0.]]] ]).astype(np.float32)


    pos, neg, loss = f(aff_pred, aff_gt, seg_gt, nhood)
    print loss
    print pos
    print '-'*40
    print neg
    print '-'*40

    g = f2(aff_pred, aff_gt, seg_gt, nhood)
    print g[0]

    
    g_true = np.array([[[[  3.,   2.,   4.,   0.,   0.,   3.],
                         [  8.,   0.,  16.,   0.,   0.,   2.],
                         [ 12.,   0.,   1.,   0.,   0.,   1.],
                         [  0.,   0.,   0.,   0.,   0.,   0.]]],


                       [[[  1.,   0.,   1.,   0.,   0.,   0.],
                         [  0.,   0.,   1.,   0.,   0.,   0.],
                         [  1.,   0.,   2.,   0.,   0.,   0.],
                         [  1.,   0.,   3.,   0.,   0.,   0.]]]])

    pos_true = np.array([[[[ 3,  2,  4,  0,  0,  3],
                           [ 8,  0, 16,  0,  0,  2],
                           [12,  0,  1,  0,  0,  1],
                           [ 0,  0,  0,  0,  0,  0]]],


                         [[[ 1,  0,  1,  0,  0,  0],
                           [ 0,  0,  1,  0,  0,  0],
                           [ 1,  0,  2,  0,  0,  0],
                           [ 1,  0,  3,  0,  0,  0]]]], dtype=np.int32)

    assert np.allclose(g[0],g_true)
    assert np.allclose(pos,pos_true)
    
    nhood = np.array([[ 0.,  1.,  0.],
                      [ 0.,  0.,  1.]])

    test_id2 = np.array([[[1, 1, 2, 2, 0, 3],
                          [1, 1, 2, 2, 0, 3],
                          [1, 1, 2, 2, 0, 3],
                          [1, 1, 2, 2, 0, 3]]], dtype=np.int32)

    aff_gt   = malis_utils.seg_to_affgraph(test_id2, nhood)
    aff_pred = np.array([ [[[ 1.,  1.,  1.,  1.,  0., 1.],
                            [ 1.,  1.,  1.,  1.,   0., 1.],
                            [ 0.9, 0.8, 1.,  1.,  0., 1.],
                            [ 0.,  0.,  0.,  0.,  0., 1.]]],

                          [[[ 1.,  0.,  1.,  0.3, 0.4,0.],
                            [ 0.7, 0.,  1.,  0.,  0., 0.],
                            [ 1.,  0.2, 1.,  0.,  0., 0.],
                            [ 1.,  0.,  1.,  0.,  0., 0.]]] ]).astype(np.float32)    