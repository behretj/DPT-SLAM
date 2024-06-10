import numpy as np
import torch
import lietorch
import droid_backends

from torch.multiprocessing import Process, Queue, Lock, Value
from collections import OrderedDict

from thirdparty.DOT.dot.models.interpolation import interpolate
from thirdparty.DOT.dot.utils.torch import get_grid

class DepthVideo:
    def __init__(self, image_size=[480, 640], buffer=400, stereo=False, device="cuda:0"):

        # current keyframe count
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]

        ### for distance measure using flow ###
        self.thresh = 2.5

        ### state attributes ###
        self.tstamp = torch.zeros(buffer, device="cuda", dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device="cuda", dtype=torch.uint8)
        self.dirty = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.red = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//4, wd//4, device="cuda", dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht//4, wd//4, device="cuda", dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device="cuda", dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device="cuda", dtype=torch.float).share_memory_()

        self.stereo = stereo
        c = 1 if not self.stereo else 2

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")

        self.image_dot = []
        
    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if (len(item) == 1): # not keyframe, only add to image_dot
            self.image_dot.append(item[0].to('cpu'))
            return
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        # self.dirty[index] = True
        self.tstamp[index] = item[0]

        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            depth = item[4][3::8,3::8]
            self.disps_sens[index] = torch.where(depth>0, 1.0/depth, depth)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.image_dot.append(item[6].to('cpu'))

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index]
                )

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric using flow between frames"""

        tstamps_list = (self.tstamp).tolist()

        sources_list = [int(tstamps_list[i]) for i in ii]
        targets_list = [int(tstamps_list[i]) for i in jj]

        d = []
        num_tracks_used = []

        for idx in range(len(sources_list)):
            ### Doing approximate flow estimation ###
            src_points = self.cotracker_track[:, sources_list[idx]].to('cuda')
            tgt_points = self.cotracker_track[:, targets_list[idx]].to('cuda')
            grid = get_grid(128, 128).to("cuda")

            if bidirectional:
                est_flow_forward, _, num_tracks_used1 = interpolate(src_points=src_points, tgt_points=tgt_points, grid=grid, version="torch3d")
                est_flow_backward, _, num_tracks_used2 = interpolate(src_points=tgt_points, tgt_points=src_points, grid=grid, version="torch3d")

                if num_tracks_used1 == 0 or num_tracks_used2 == 0:
                    num_tracks_used.append(0)
                    d.append(0)
                    continue

                num_tracks_used.append((num_tracks_used1 + num_tracks_used2) / 2)

                d1 = est_flow_forward.norm(dim=-1).mean().item()
                d2 = est_flow_backward.norm(dim=-1).mean().item()

                d.append(.5 * (d1 + d2))
            else:
                est_flow, _, num_tracks = interpolate(src_points=src_points, tgt_points=tgt_points, grid=grid, version="torch3d")

                if num_tracks_used == 0:
                    num_tracks_used.append(0)
                    d.append(0)
                    continue

                d.append(est_flow.norm(dim=-1).mean().item())
                num_tracks_used.append(num_tracks)
        
        d = torch.tensor(d).to('cuda')
        num_tracks_used = torch.tensor(num_tracks_used).to('cuda')
        return d, num_tracks_used


    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False):
        """ dense bundle adjustment (DBA) """

        with self.get_lock():

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.disps_sens,
                target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)

            self.disps.clamp_(min=0.001)