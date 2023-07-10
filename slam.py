#!/usr/bin/env python
import sys
import time
import numpy as np
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
import cv2
from display import Display3D
from frame import Frame, match_frames
from pointmap import Map, Point
from helpers import triangulate, add_ones, saturation

np.set_printoptions(suppress=True)

class SLAM(object):
    def __init__(self, W, H, K, algorithm = 'ORB', frame_step=5):
        # main classes
        self.mapp = Map()
        self.image = None
        # params
        self.W, self.H, self.K = W, H, K
        self.algorithm = algorithm
        self.frame_step = frame_step

    def process_frame(self, img):
        self.image = img
        start_time = time.time()
        assert self.image.shape[0:2] == (self.H, self.W)
        frame = Frame(self.mapp, self.image, self.K, verts=None, algorithm = self.algorithm)

        if frame.id == 0:
            return

        f1 = self.mapp.frames[-1]
        f2 = self.mapp.frames[-2]

        idx1, idx2, Rt = match_frames(f1, f2)

        # add new observations if the point is already observed in the previous frame
        # TODO: consider tradeoff doing this before/after search by projection
        for i, idx in enumerate(idx2):
            if f2.pts[idx] is not None and f1.pts[idx1[i]] is None:
                f2.pts[idx].add_observation(f1, idx1[i])

        # get initial positions from fundamental matrix
        f1.pose = Rt @ f2.pose

        # pose optimization
        pose_opt = self.mapp.optimize(local_window=1, fix_points=True)
        sbp_pts_count = 0

        # search by projection
        if len(self.mapp.points) > 0:
            # project *all* the map points into the current frame
            map_points = np.array([p.homogeneous() for p in self.mapp.points])
            projs = (self.K @ f1.pose[:3] @ map_points.T).T
            projs = projs[:, 0:2] / projs[:, 2:]

            # only the points that fit in the frame
            good_pts = (projs[:, 0] > 0) & (projs[:, 0] < self.W) & \
                                 (projs[:, 1] > 0) & (projs[:, 1] < self.H)

            for i, p in enumerate(self.mapp.points):
                if not good_pts[i] or f1 in p.frames:
                    # point not visible in frame
                    # we already matched this map point to this frame
                    # TODO: understand this better
                    continue
                for m_idx in f1.kd.query_ball_point(projs[i], 2):
                    # if point unmatched
                    if f1.pts[m_idx] is None and p.orb_distance(f1.des[m_idx]) < 64.0:
                        p.add_observation(f1, m_idx)
                        sbp_pts_count += 1
                        break

        # triangulate the points we don't have matches for
        good_pts4d = np.array([f1.pts[i] is None for i in idx1])

        # do triangulation in global frame
        pts4d = triangulate(f1.pose, f2.pose, f1.kps[idx1], f2.kps[idx2])
        good_pts4d &= np.abs(pts4d[:, 3]) != 0
        pts4d /= pts4d[:, 3:]       # homogeneous 3-D coords

        # adding new points to the map from pairwise matches
        new_pts_count = 0
        for i, p in enumerate(pts4d):
            if not good_pts4d[i]:
                continue

            # check points are in front of both cameras
            pl1 = f1.pose @ p
            pl2 = f2.pose @ p
            if pl1[2] < 0 or pl2[2] < 0:
                continue

            # reproject
            pp1 = self.K @ pl1[:3]
            pp2 = self.K @ pl2[:3]

            # check reprojection error
            pp1 = (pp1[0:2] / pp1[2]) - f1.key_pts[idx1[i]]
            pp2 = (pp2[0:2] / pp2[2]) - f2.key_pts[idx2[i]]
            pp1, pp2 = (np.sum(pp1**2), np.sum(pp2**2))
            if pp1 > 2 or pp2 > 2:
                continue

            # color points from frame
            cx = np.int32(f1.key_pts[idx1[i],0])
            cy = np.int32(f1.key_pts[idx1[i],1])
            color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[cy, cx]

            pt = Point(self.mapp, p[0:3], color)
            pt.add_observation(f2, idx2[i])
            pt.add_observation(f1, idx1[i])
            new_pts_count += 1

        print("Adding:   %d new points, %d search by projection" % (new_pts_count, sbp_pts_count))

        # optimize the map
        # if frame.id >= 4:
        if frame.id >= 2 and frame.id % self.frame_step == 0:
            err = self.mapp.optimize() #verbose=True)
            print("Optimize: %f units of error" % err)

        print("Map:      %d points, %d frames" % (len(self.mapp.points), len(self.mapp.frames)))
        print("Time:     %.2f ms" % ((time.time()-start_time)*1000.0))
        print(np.linalg.inv(f1.pose))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("%s <video.mp4>" % sys.argv[0])
        exit(-1)

    disp3d = Display3D()

    cap = cv2.VideoCapture(sys.argv[1])

    # camera parameters
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    CNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    F = 225

    if W > 1024:
        downscale = 1024.0/W
        F *= downscale
        H = int(H * downscale)
        W = 1024
    print("using camera %dx%d with F %f" % (W,H,F))

    # camera intrinsics
    K = np.array([[F, 0, W//2],[0, F, H//2],[0, 0, 1]])
    # Kinv = np.linalg.inv(K)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 2450)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 3000)
    cv2.namedWindow('SLAM', cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow('SLAM', cv2.WINDOW_GUI_EXPANDED | cv2.WINDOW_AUTOSIZE)
    slam = SLAM(W, H, K, algorithm = 'AKAZE', frame_step=4)

    frame_step = 0
    frame_scale = 0.75
    while cap.isOpened():
        ret, frame = cap.read()
        frame_counter = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if frame_step < 4:
            frame_step +=1
            continue
        frame_step = 0
        frame = saturation(cv2.resize(frame, (W, H)), 1.2)
        print('\n*** frame {}/{} ***'.format(frame_counter, CNT))

        slam.process_frame(frame)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break
        elif key == ord('p'):
            cv2.waitKey(-1)

        # 3D display
        disp3d.paint(slam.mapp)

        # 2D display
        img = slam.mapp.frames[-1].annotate(frame)
        img = cv2.resize(img, (int(W*frame_scale), int(H*frame_scale)))
        cv2.imshow('SLAM', img)

    cv2.destroyAllWindows()
