# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:07:37 2015

@author: Marius Felix Killinger
"""
import os
import numpy as np
import theano
from theano import gof
import theano.tensor as T
from theano.gradient import disconnected_type

from malis import get_malis_weights


class Malis(theano.Op):
    """
    MALIS
    
    affinity_pred = affinity, edge probability
    """
    __props__ = ()

    def make_node(self, *inputs):
      inputs = map(T.as_tensor_variable, inputs) 
      affinity_pred, affinity_gt, nhood = inputs
      
      if affinity_pred.ndim!=4:
          raise ValueError("affinity_pred must be convertable to a\
          TensorVariable of dimensionality 4. This one has ndim=%i"\
          %affinity_pred.ndim)
 
      if affinity_gt.ndim!=4:
          raise ValueError("affinity_gt must be convertable to a\
          TensorVariable of dimensionality 4. This one has ndim=%i"\
          %affinity_gt.ndim)
          
      if nhood.ndim!=2:
          raise ValueError("nhood must be convertable to a\
          TensorVariable of dimensionality 2. This one has ndim=%i"\
          %nhood.ndim)          
     
      malis_weights_pos = T.TensorType(
                        dtype='int32',
                        broadcastable=(False,)*4)()
                        
      malis_weights_neg = T.TensorType(
                        dtype='int32',
                        broadcastable=(False,)*4)()
                        
      labels_gt = T.TensorType(
                        dtype='int32',
                        broadcastable=(False,)*3)()                        
               
      outputs = [malis_weights_pos, malis_weights_neg, labels_gt]               

      return gof.Apply(self, inputs, outputs)

    def perform(self, node, inputs, output_storage):
      """
      Inputs
      ------
      
      affinity_pred: (edges, z, x, y) or (edges, x, y)
      nhood: (edges, n_dim)
      threshold: float
      
      Outputs
      -------
      
      weight matrix, locally constant wrt. to input --> grad==0
      
      
      Outline
      -------
      
      - Computes for all pixel-pairs the MaxiMin-Affinity
      - Separately for pixel-pairs that should/should not be connected
      - Every time an affinity prediction is a MaxiMin-Affinity its weight is
        incremented by one in the output matrix (in different slices depending
        on whether that that pair should/should not be connected)
      - Additionally the number of correct/incorret connections wrt to threshold
        are collected
      """
      affinity_pred, affinity_gt, nhood = inputs
      pos, neg, gt = output_storage   
      pos[0], neg[0], gt[0] = get_malis_weights(affinity_pred,
                                                affinity_gt,
                                                nhood)

    def grad(self, inputs, outputs_gradients):
      # The gradient of all outputs is 0 w.r.t to all inputs
      return [disconnected_type(), disconnected_type(), disconnected_type()]     

    def connection_pattern(self, node):
      # The gradient of all outputs is 0 w.r.t to all inputs
      return [[False, False, False],[False, False, False],[False, False, False]]
      
      
malis_weights = Malis()

if __name__=="__main__":
    import theano.sandbox.cuda
    theano.sandbox.cuda.use("gpu0")
    import theano
    from malis import seg_to_affgraph
    
    print "TESTING/DEMO:"

    
    aff_pred_t = T.TensorType('float32', [False,]*4, name='aff_pred')()
    aff_gt_t   = T.TensorType('int32',   [False,]*4, name='aff_gt')()
    neigh_t    = T.TensorType('int32',   [False,]*2, name='neighb')()
    
    
    pos_t, neg_t, seg = malis_weights(aff_pred_t, aff_gt_t, neigh_t)    
    loss_t = T.sum(pos_t * aff_pred_t)
    
    f = theano.function([aff_pred_t, aff_gt_t, neigh_t], [pos_t, neg_t, seg, loss_t]) 
    grad_t = theano.grad(loss_t, aff_pred_t)
    f2 = theano.function([aff_pred_t, aff_gt_t, neigh_t], [grad_t,]) 
    
    nhood = np.array([[ 0.,  1.,  0.],
                      [ 0.,  0.,  1.]], dtype=np.int32) 
                        
    test_id2 = np.array([[[1, 1, 2, 2, 0, 3],
                          [1, 1, 2, 2, 0, 3],
                          [1, 1, 2, 2, 0, 3],
                          [1, 1, 2, 2, 0, 3]]], dtype=np.int32)
                      
    aff_gt   = seg_to_affgraph(test_id2, nhood)                  
    aff_pred = np.array([ [[[ 1.,  1.,  1.,  1.,  0., 1.],
                            [ 1.,  1.,  1.,  1.,   0., 1.],
                            [ 0.9, 0.8, 1.,  1.,  0., 1.],
                            [ 0.,  0.,  0.,  0.,  0., 1.]]],
                          
                          [[[ 1.,  0.,  1.,  0.3, 0.4,0.],
                            [ 0.7, 0.,  1.,  0.,  0., 0.],
                            [ 1.,  0.2, 1.,  0.,  0., 0.],
                            [ 1.,  0.,  1.,  0.,  0., 0.]]] ]).astype(np.float32)


    pos, neg, seg, loss = f(aff_pred, aff_gt, nhood)
    print loss
    print pos
    print '-'*40
    print neg
    print '-'*40

    g = f2(aff_pred, aff_gt, nhood)
    print g[0]  