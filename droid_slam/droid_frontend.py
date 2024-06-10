import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph


class DroidFrontend:
    def __init__(self, video, args):
        self.video = video
        self.graph = FactorGraph(video, max_factors=48, upsample=args.upsample)

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.count = 0
        self.init_iters = 2

        self.max_age = 25

        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius

    def __update(self):
        """ add edges, perform update """

        self.count += 1
        self.t1 += 1

        # 1 remove old pairs of frames from graph
        self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        # 2 add frame pairs to graph based on distance measure (for newly added keyframe)
        #   compute refined flow and weights for new pairs in graph
        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0), 
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)

        self.video.disps[self.t1-1] = torch.where(self.video.disps_sens[self.t1-1] > 0, 
           self.video.disps_sens[self.t1-1], self.video.disps[self.t1-1])

        # 3 do BA to refine poses and depth maps
        self.graph.update_DOT_SLAM(None, None, use_inactive=True)

        # set initial pose for next frame
        poses = SE3(self.video.poses)
        d, _ = self.video.distance([self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True)

        if d.item() < self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 2)
            
            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        else:
            # 4 do BA to refine poses and depth maps
            self.graph.update_DOT_SLAM(None, None, use_inactive=True)

        # set pose for next iteration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        # update visualization
        self.video.dirty[self.graph.ii.min():self.t1] = True

    def __initialize(self):
        """ initialize the SLAM system """

        self.t0 = 0
        self.t1 = self.video.counter.value

        # 1 add frame pairs to graph based on neighborhood
        #   compute refined flow and weights for new pairs in graph
        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        # 2 do BA to refine poses and depth maps
        self.graph.update_DOT_SLAM(1, use_inactive=True, ba_calls=4*self.init_iters)

        # 3 add frame pairs to graph based on distance measure
        #   compute refined flow and weights for new pairs in graph
        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False, init=True)

        # 4 do BA to refine poses and depth maps
        self.graph.update_DOT_SLAM(1, use_inactive=True, ba_calls=4*self.init_iters)

        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone()
        self.last_disp = self.video.disps[self.t1-1].clone()
        self.last_time = self.video.tstamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[:self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    def __call__(self):
        """ main update """

        # do initialization
        if not self.is_initialized and self.video.counter.value >= self.warmup:
            self.__initialize()
            
        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()
