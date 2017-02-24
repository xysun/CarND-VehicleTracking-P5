import numpy as np

def get_hog_feature_parallel(pos,hogs,nblocks_per_window):
    (xpos,ypos,idx) = pos
    hog = hogs[idx]
    hog_feat1 = hog[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
    hog_features = np.hstack((hog_feat1))
    return hog_features
