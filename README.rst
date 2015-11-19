MALIS 
=====

Structured loss function for supervised learning of segmentation and clustering
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Python wrapper for C++ functions for computing the MALIS loss.
In this implementation the affinity prediction, the affinity ground truth and the segmentation ground truth (label IDs) are passed to ``malis.get_malis_weights``. This function returns two weight/count matrices which quantify the impact of the predicted affinities: one for affinities that should be 1 (pos) and one for affinities that should be 0 (neg). These counts are intended to weight the "pixelwise" errors of the affinity prediction.

This fork additionally includes a ready to use Theano op that returns weighting matrices. Example usage: ::


    pos_count, neg_count = malis.malisop.malis_weights(affinity_pred, aff_gt, seg_gt, neigh_pattern)
      
    
    weighted_pos = -T.xlogx.xlogy0(pos_count, affinity_pred) # drive up prediction for "connected" here
    weighted_neg = -T.xlogx.xlogy0(neg_count, disconnect_pred) # drive up prediction for "disconnected" here    
    loss = T.sum(weighted_pos + weighted_neg)


The MALIS loss is described here:

SC Turaga, KL Briggman, M Helmstaedter, W Denk, HS Seung (2009). *Maximin learning of image segmentation*. _Advances in Neural Information Processing Systems (NIPS) 2009_.

http://papers.nips.cc/paper/3887-maximin-affinity-learning-of-image-segmentation

Installation
============

For installation as python package: download and run inside directory

	pip install . (pip-install options)

(or use URL of github repo instead of '.' to let pip download the files)


Building c++ extension only: download and run inside directory

	./make.sh
