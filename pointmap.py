# Point Map

from helpers import poseRt, hamming_distance, add_ones
from constants import CULLING_ERR_THRES
from frame import Frame
import time
import numpy as np
# np.finfo(np.dtype("float32"))
# np.finfo(np.dtype("float64"))
import g2o

from optimize_g2o import optimize

LOCAL_WINDOW = 20
#LOCAL_WINDOW = None

class Point(object):
    # A Point is a 3-D point in the world
    # Each Point is observed in multiple Frames

    def __init__(self, mapp, loc, color, tid=None):
        self.pt = np.array(loc)
        self.frames = []
        self.idxs = []
        self.color = np.copy(color)
        self.id = tid if tid is not None else mapp.add_point(self)

    def homogeneous(self):
        return add_ones(self.pt)

    def orb(self):
        return [f.des[idx] for f,idx in zip(self.frames, self.idxs)]

    def orb_distance(self, des):
        return min([hamming_distance(o, des) for o in self.orb()])

    def delete(self):
        for f,idx in zip(self.frames, self.idxs):
            f.pts[idx] = None
        del self

    def add_observation(self, frame, idx):
        assert frame.pts[idx] is None
        assert frame not in self.frames
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)

class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.max_frame = 0
        self.max_point = 0

    def add_point(self, point):
        ret = self.max_point
        self.max_point += 1
        self.points.append(point)
        return ret

    def add_frame(self, frame):
        ret = self.max_frame
        self.max_frame += 1
        self.frames.append(frame)
        return ret

    # Optimizer
    def optimize(self, local_window=LOCAL_WINDOW, fix_points=False, verbose=False, rounds=50):
        err = optimize(self.frames, self.points, local_window, fix_points, verbose, rounds)

        # prune points
        culled_pt_count = 0
        for p in self.points:
            # <= 4 match point that's old
            old_point = len(p.frames) <= 4 and p.frames[-1].id+7 < self.max_frame

            # compute reprojection error
            errs = []
            for f, idx in zip(p.frames, p.idxs):
                uv = f.kps[idx]
                proj = f.pose[:3] @ p.homogeneous()
                proj = proj[0:2] / proj[2]
                errs.append(np.linalg.norm(proj-uv))

            # cull
            if old_point or np.mean(errs) > CULLING_ERR_THRES:
                culled_pt_count += 1
                self.points.remove(p)
                p.delete()
        print("Culled:   %d points" % (culled_pt_count))
        # print("Optimize: %f units of error" % err)
        return err

