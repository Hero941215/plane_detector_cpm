# plane_detector_cpm

Implementation of the fast plane detector as described in the paper "Fast  Plane  Detection  and  Polygonalization  in  noisy  3D  Range  Images", of Poppinga et al. (IROS 2008)  https://ieeexplore.ieee.org/abstract/document/4650729

# add
In the candidate_segment class, scan smoothness is used to preprocess range image for deleting some edge points.

We add the closest point merge function in "plane_detector" to handle the occlusion problem in the region growth detection algorithm.

