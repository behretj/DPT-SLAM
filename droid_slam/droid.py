import torch
import lietorch
import numpy as np

from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process

import json


class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.video, self.args)

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.video)


    """
    main thread: processes all frames sequentially
    """
    def track(self, tstamp, image, depth=None, intrinsics=None, image_dot=None):

        with torch.no_grad():
            self.filterx.track_buffer(tstamp, image, depth, intrinsics, image_dot, self.frontend)


    """
    method to terminate frontend process and return poses also for non-keyframes
    """
    def terminate(self, stream=None):

        del self.frontend

        torch.cuda.empty_cache()

        camera_trajectory = self.traj_filler(stream)

        return camera_trajectory.inv().data.cpu().numpy()
