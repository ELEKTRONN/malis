# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:47:59 2015

@author: Marius Felix Killinger
"""

import numpy as np
import matplotlib.pyplot as plt

import malis
from elektronn.training.trainutils import timeit 
import time

  
if __name__=="__main__":     
    test_bmap = np.array([[[0, 0, 1, 0],
                           [0, 0, 1, 0],
                           [0, 0, 1, 0],
                           [0, 0, 1, 0],
                           [0, 0, 1, 0]],
                    
                          [[0, 1, 0, 0],
                           [0, 1, 1, 0],
                           [0, 0, 1, 0],
                           [0, 0, 1, 0],
                           [0, 0, 1, 0]],
                    
                          [[0, 1, 0, 0],
                           [0, 1, 0, 0],
                           [0, 1, 1, 0],
                           [0, 0, 1, 0],
                           [0, 0, 1, 0]]])

    test_ids = np.array([
                       [[3, 3, 0, 1],
                        [3, 3, 0, 1],
                        [3, 3, 0, 1],
                        [3, 3, 0, 1],
                        [3, 3, 0, 1]],
                
                       [[3, 0, 1, 1],
                        [3, 0, 0, 1],
                        [3, 3, 0, 1],
                        [3, 3, 0, 1],
                        [3, 3, 0, 1]],
                
                       [[3, 0, 1, 1],
                        [3, 0, 1, 1],
                        [3, 0, 0, 1],
                        [3, 3, 0, 1],
                        [3, 3, 0, 1]]])
          
    
    
                    
#    extended_2d = np.array([[0,2,0],[0,0,2],[0,1,1]], dtype=np.int)
#    nhood_ext = np.vstack([nhood, extended_2d])
    # 2D Test
    nhood = np.array([[ 0.,  1.,  0.],
                        [ 0.,  0.,  1.]]) 
                        
#    test_id2 = np.array([[[1, 1, 2, 2, 0],
#                          [1, 1, 2, 2, 0],
#                          [1, 1, 2, 2, 0],
#                          [1, 1, 2, 2, 0]]], dtype=np.int32)
#                      
#    aff_gt   = malis.seg_to_affgraph(test_id2, nhood)                  
#    aff_pred = np.array([ [[[ 1.,  1.,  1.,  1.,  0.],
#                            [ 0.5, 1.,  1.,  1.,  0.],
#                            [ 1.,  1.,  1.,  1.,  0.],
#                            [ 0.,  0.,  0.,  0.,  0.]]],
#                          
#                          [[[ 1.,  0.,  1.,  0.,  0.],
#                            [ 1.,  0.,  1.,  0.5, 0.],
#                            [ 1.,  0.5, 1.,  0.,  0.],
#                            [ 1.,  0.,  1.,  0.,  0.]]] ]).astype(np.float32)
#

    nhood = np.array([[ 0.,  1.,  0.],
                      [ 0.,  0.,  1.]]) 
                        
    test_id2 = np.array([[[1, 1, 2, 2, 0, 3],
                          [1, 1, 2, 2, 0, 3],
                          [1, 1, 2, 2, 0, 3],
                          [1, 1, 2, 2, 0, 3]]], dtype=np.int32)
                      
    aff_gt   = malis.seg_to_affgraph(test_id2, nhood)                  
    aff_pred = np.array([ [[[ 1.,  1.,  1.,  1.,  0., 1.],
                            [ 1.,  1.,  1.,  1.,   0., 1.],
                            [ 0.9, 0.8, 1.,  1.,  0., 1.],
                            [ 0.,  0.,  0.,  0.,  0., 1.]]],
                          
                          [[[ 1.,  0.,  1.,  0.3, 0.4,0.],
                            [ 0.7, 0.,  1.,  0.,  0., 0.],
                            [ 1.,  0.2, 1.,  0.,  0., 0.],
                            [ 1.,  0.,  1.,  0.,  0., 0.]]] ]).astype(np.float32)


#    test_id2 = np.array([[[1, 1, 2, ],
#                          [1, 1, 2, ],
#                          [1, 1, 2, ],
#                          [1, 1, 2, ]]], dtype=np.int32)
#                      
#    aff_gt   = malis.seg_to_affgraph(test_id2, nhood)                  
#    aff_pred = np.array([ [[[ 1.,  1.,  1.],
#                            [ 0.9, 0.8, 1.],
#                            [ 1.,  1.,  1.],
#                            [ 0.,  0.,  0.]]],
#                          
#                          [[[ 1.,  0.,  0],
#                            [ 0.7, 0.,  0],
#                            [ 1.,  0.2, 0],
#                            [ 1.,  0.,  0]]] ]).astype(np.float32)                                
    pos_counts, neg_counts, seg  = malis.get_malis_weights(aff_pred, aff_gt, nhood)
    print pos_counts
    print '-'*40
    print neg_counts      


# TIMING
    large_aff = np.random.rand(3,50,100,100).astype(np.float32)
    large_gt  = np.random.rand(3,50,100,100).astype(np.float32) 
    neigh_nhood  = np.eye(3).astype(np.int32)
    
    t = time.time()         
    for i in xrange(4):
      ret = malis.get_malis_weights(large_aff,
                            large_gt, neigh_nhood,)
    dt = time.time() - t
    print dt/4              