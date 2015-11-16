# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:47:59 2015

@author: Marius Felix Killinger
"""

import numpy as np
import matplotlib.pyplot as plt

import malis
from elektronn.training.trainutils import timeit 

  
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
          
    pattern = np.array([[ 0.,  1.,  0.],
                        [ 0.,  0.,  1.]])
                        
#    extended_2d = np.array([[0,2,0],[0,0,2],[0,1,1]], dtype=np.int)
#    pattern_ext = np.vstack([pattern, extended_2d])
    
          
    test_id2 = np.array([[[1, 1, 2, 2, 0],
                          [1, 1, 2, 2, 0],
                          [1, 1, 2, 2, 0],
                          [1, 1, 2, 2, 0]]], dtype=np.int32)
                      
    aff_gt   = malis.seg_to_affgraph(test_id2, pattern)                  
    aff_pred = np.array([ [[[ 1.,  1.,  1.,  1.,  0.],
                            [ 0.5, 0.,  1.,  1.,  0.],
                            [ 1.,  1.,  1.,  1.,  0.],
                            [ 0.,  0.,  0.,  0.,  0.]]],
                          
                          [[[ 1.,  0.,  1.,  0.,  0.],
                            [ 1.,  0.5, 1.,  0.,  0.],
                            [ 1.,  0.,  1.,  0.,  0.],
                            [ 1.,  0.,  1.,  0.,  0.]]] ]).astype(np.float32)
                                
    pos_counts, neg_counts, seg  = malis.get_malis_weights(aff_pred, aff_gt, pattern)
    print seg.reshape(test_id2.shape)
    pos_counts, neg_counts, seg  = malis.get_malis_weights(aff_pred, aff_gt, pattern)                           