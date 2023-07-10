import os
import cv2
import numpy as np
# np.finfo(np.dtype("float32"))
# np.finfo(np.dtype("float64"))
from scipy.spatial import cKDTree
from constants import RANSAC_RESIDUAL_THRES, RANSAC_MAX_TRIALS
np.set_printoptions(suppress=True)

from skimage.measure import ransac
from helpers import fundamentalToRt, normalize, EssentialMatrixTransform

def featureMappingORB(frame):
    orb = cv2.ORB_create()
    # pts = cv2.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8), 1000, qualityLevel=0.01, minDistance=7)
    pts = cv2.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8), 1000, qualityLevel=0.01, minDistance=7)
    key_pts = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
    key_pts, descriptors = orb.compute(frame, key_pts)
    # Return Key_points and ORB_descriptors
    return np.array([(kp.pt[0], kp.pt[1]) for kp in key_pts]), descriptors

def featureMappingAKAZE(frame):
    detect = cv2.AKAZE_create()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    key_pts, des = detect.detectAndCompute(frame_gray, None)
    return np.array([(kp.pt[0], kp.pt[1]) for kp in key_pts]), des

def SLAMBFMatcher(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    return bf.knnMatch(des1.des, des2.des, k=2)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# matcher_function = {'BF':SLAMBFMatcher, 'Flann':SLAMFlannBasedMatcher}
def match_frames(f1, f2):
    matches = bf.knnMatch(f1.des, f2.des, k=2)
    # Lowe's ratio test
    ret = []
    idx1, idx2 = [], []
    idx1s, idx2s = set(), set()
    for m, n in matches:
        # be within orb distance 32
        if m.distance < 0.75*n.distance and m.distance < 32:
            p1 = f1.kps[m.queryIdx]
            p2 = f2.kps[m.trainIdx]
            # keep around indices
            # TODO: refactor this to not be O(N^2)
            if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                idx1s.add(m.queryIdx)
                idx2s.add(m.trainIdx)
                ret.append((p1, p2))

    # no duplicates
    assert(len(set(idx1)) == len(idx1))
    assert(len(set(idx2)) == len(idx2))

    assert len(ret) >= 8
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    # fit matrix
    model, inliers = ransac((ret[:, 0], ret[:, 1]),
                            EssentialMatrixTransform,
                            min_samples=8,
                            residual_threshold=RANSAC_RESIDUAL_THRES,
                            max_trials=RANSAC_MAX_TRIALS)
    print("Matches:  %d -> %d -> %d -> %d" % (len(f1.des), len(matches), len(inliers), sum(inliers)))
    return idx1[inliers], idx2[inliers], fundamentalToRt(model.params)

def show_attributes(frame, attribut):
    cv2.rectangle(frame, (30, 0), (110, 45), (110,50,30), -1)
    cv2.putText(frame, attribut, (45, 30), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255,255,255), 1)

feature_mapping = {'ORB':featureMappingORB, 'AKAZE':featureMappingAKAZE}

class Frame(object):
    def __init__(self, mapp, img, K, pose=np.eye(4), tid=None, verts=None, algorithm='ORB'):
        self.K = np.array(K)
        self.pose = np.array(pose)
        self.algorithm = algorithm

        self.h, self.w = img.shape[0:2]
        self.key_pts, self.des = feature_mapping[algorithm](img)
        self.pts = [None]*len(self.key_pts)
        self.id = tid if tid is not None else mapp.add_frame(self)

    def annotate(self, img):
        # paint annotations on the image
        for pt in self.key_pts:
            u1, v1 = np.int32(pt[0]), np.int32(pt[1])
            cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=6)
            cv2.drawMarker(img, (u1, v1), (0, 255, 255), 1, 8, 1, 8)
            show_attributes(img, self.algorithm)
        return img

    # inverse of intrinsics matrix
    @property
    def Kinv(self):
        if not hasattr(self, '_Kinv'):
            self._Kinv = np.linalg.inv(self.K)
        return self._Kinv

    # normalized keypoints
    @property
    def kps(self):
        if not hasattr(self, '_kps'):
            self._kps = normalize(self.Kinv, self.key_pts)
        return self._kps

    # KD tree of unnormalized keypoints
    @property
    def kd(self):
        if not hasattr(self, '_kd'):
            self._kd = cKDTree(self.key_pts)
        return self._kd

