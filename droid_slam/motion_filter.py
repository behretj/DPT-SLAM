import cv2
import torch
import lietorch

from collections import OrderedDict
from collections import deque

import geom.projective_ops as pops
import torch.nn.functional as F

from thirdparty.DOT.dot.models.point_tracking import PointTracker
from thirdparty.DOT.dot.models.interpolation import interpolate
from thirdparty.DOT.dot.utils.torch import get_grid

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib import colormaps
from tqdm import tqdm

class MotionFilter:
    """ This class is used to filter incoming frames """

    def __init__(self, video, thresh=2.5, device="cuda:0",
                    tracker_config="thirdparty/DOT/configs/cotracker2_patch_4_wind_8.json",
                    tracker_path="thirdparty/DOT/checkpoints/movi_f_cotracker2_patch_4_wind_8.pth",
                    estimator_config="thirdparty/DOT/configs/raft_patch_8.json",
                    estimator_path="thirdparty/DOT/checkpoints/cvo_raft_patch_8.pth"):

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        tracker_height, tracker_width = 512, 512
        self.online_point_tracker = PointTracker(tracker_height, tracker_width, tracker_config, tracker_path, estimator_config, estimator_path, isOnline=True).to('cuda')
        self.target_batch_size = 4
        self.buffer = deque(maxlen=2*self.target_batch_size)
        self.droid_buffer = deque(maxlen=2*self.target_batch_size)
        self.last_tstamp = None
        self.is_first_step = True


    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track_buffer(self, tstamp, image, depth=None, intrinsics=None, image_dot=None, frontend=None):
        self.buffer.append(image_dot.to('cuda'))
        self.droid_buffer.append((tstamp, image, depth, intrinsics, image_dot.to('cuda')))
        
        if (tstamp + 1) % self.target_batch_size == 0:
            data = {}
            data["video_chunk"] = torch.stack(list(self.buffer), dim=1).permute(1, 0, 2, 3)[None]   # video =(Batch, frames, channel, height, width)
            B, T, C, h, w = data["video_chunk"].shape

            H, W = 512,512 #self.resolution
            if h != H or w != W: #Reshape the frames to RAFT input size (512x512)
                data["video_chunk"] = data["video_chunk"].reshape(B * T, C, h, w)
                data["video_chunk"] = F.interpolate(data["video_chunk"], size=(H, W), mode="bilinear")
                data["video_chunk"] = data["video_chunk"].reshape(B, T, C, H, W)
            ## initilization of tracks
            self.video.cotracker_track = self.online_point_tracker(data, mode="tracks_online_droid")["tracks"]
            if self.target_batch_size<=tstamp:
                self.video.cotracker_track = torch.stack([self.video.cotracker_track[..., 0] / (w - 1), self.video.cotracker_track[..., 1] / (h - 1), self.video.cotracker_track[..., 2]], dim=-1).to('cpu')
                # all the images have been registered in CoTracker, we can add them to Droid now:
                for args in self.droid_buffer:
                    self.track(*args)
                    frontend()
                self.droid_buffer.clear()


    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None, image_dot=None):
        """ main update operation - run on every frame in video """

        Id = lietorch.SE3.Identity(1,).data.squeeze()

        # normalize images
        inputs = image[None, :, [2,1,0]].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            self.video.append(tstamp, image[0], Id, 1.0, depth, intrinsics / 4.0, image_dot)
            self.last_tstamp = tstamp

        ### only add new frame if there is enough motion ###
        else:
            ### Doing approximate flow estimation
            src_points = self.video.cotracker_track[:, self.last_tstamp].to('cuda')
            tgt_points = self.video.cotracker_track[:, tstamp].to('cuda')
            grid = get_grid(128, 128).to("cuda")

            est_flow, _, _ = interpolate(src_points=src_points, tgt_points=tgt_points, grid=grid, version="torch3d")

            # check motion magnitude / add new frame to video
            if est_flow.norm(dim=-1).mean().item() > self.thresh:
                self.count = 0
                self.video.append(tstamp, image[0], None, None, depth, intrinsics / 4.0, image_dot)
                self.last_tstamp = tstamp
            else:
                self.video.append(image_dot)
                self.count += 1

        torch.cuda.empty_cache()
