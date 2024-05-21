import cv2
import torch
import lietorch

from collections import OrderedDict
from droid_net import DroidNet
from collections import deque

import geom.projective_ops as pops
from modules.corr import CorrBlock
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
    """ This class is used to filter incoming frames and extract features """

    # def __init__(self, net, video, thresh=2.5, device="cuda:0",
    def __init__(self, video, thresh=2.5, device="cuda:0",
                    tracker_config="thirdparty/DOT/configs/cotracker2_patch_4_wind_8.json",
                    tracker_path="thirdparty/DOT/checkpoints/movi_f_cotracker2_patch_4_wind_8.pth",
                    estimator_config="thirdparty/DOT/configs/raft_patch_8.json",
                    estimator_path="thirdparty/DOT/checkpoints/cvo_raft_patch_8.pth"):
        
        # split net modules
        # self.cnet = net.cnet
        # self.fnet = net.fnet
        # self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        # self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        # self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        tracker_height, tracker_width = 512, 512
        self.online_point_tracker = PointTracker(tracker_height, tracker_width, tracker_config, tracker_path, estimator_config, estimator_path, isOnline=True).to('cuda')
        self.target_batch_size = 4
        self.buffer = deque(maxlen=2*self.target_batch_size)
        self.droid_buffer = deque(maxlen=2*self.target_batch_size)
        self.last_tstamp = None
        self.is_first_step = True

    # @torch.cuda.amp.autocast(enabled=True)
    # def __context_encoder(self, image):
    #     """ context features """
    #     net, inp = self.cnet(image).split([128,128], dim=2)
    #     return net.tanh().squeeze(0), inp.relu().squeeze(0)

    # @torch.cuda.amp.autocast(enabled=True)
    # def __feature_encoder(self, image):
    #     """ features for correlation volume """
    #     return self.fnet(image).squeeze(0)


    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track_buffer(self, tstamp, depth=None, intrinsics=None, image_dot=None, frontend=None):
    # def track_buffer(self, tstamp, image, depth=None, intrinsics=None, image_dot=None, frontend=None):
        self.buffer.append(image_dot.to('cuda'))
        self.droid_buffer.append((tstamp, depth, intrinsics, image_dot.to('cuda')))
        # self.droid_buffer.append((tstamp, image, depth, intrinsics, image_dot.to('cuda')))
        
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
            self.video.cotracker_track = self.online_point_tracker(data, mode="tracks_at_motion_boundaries_online_droid")["tracks"]
            if self.target_batch_size<=tstamp:
                # TODO: Do the tracks get divided every iteration making them super small?
                self.video.cotracker_track = torch.stack([self.video.cotracker_track[..., 0] / (w - 1), self.video.cotracker_track[..., 1] / (h - 1), self.video.cotracker_track[..., 2]], dim=-1).to('cpu')
                # all the images have been registered in CoTracker, we can add them to Droid now:
                for args in self.droid_buffer:
                    self.track(*args)
                    # TODO: think about this (probably bad practice!)
                    frontend()
                self.droid_buffer.clear()

                if len(self.video.cotracker_track[0]) == 380:
                    self.plot_traj_video()

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, depth=None, intrinsics=None, image_dot=None):
    # def track(self, tstamp, image, depth=None, intrinsics=None, image_dot=None):
        """ main update operation - run on every frame in video """

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        # ht = image.shape[-2] // 8
        # wd = image.shape[-1] // 8

        # normalize images
        # inputs = image[None, :, [2,1,0]].to(self.device) / 255.0
        # inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        # gmap = self.__feature_encoder(inputs)

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            # net, inp = self.__context_encoder(inputs[:,[0]])
            # self.net, self.inp, self.fmap = net, inp, gmap
            self.video.append(tstamp, Id, 1.0, depth, intrinsics / 4.0, image_dot)
            # self.video.append(tstamp, image[0], Id, 1.0, depth, intrinsics / 4.0, gmap, net[0,0], inp[0,0], image_dot)
            self.last_tstamp = tstamp

        ### only add new frame if there is enough motion ###
        else:                
            #### NOT needed anymore
            # coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
            # corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            # _, delta, weight = self.update(self.net[None], self.inp[None], corr)
            
            
            ### "Doing approximate flow estimation, replacing the above line:
            src_points = self.video.cotracker_track[:, self.last_tstamp].to('cuda')
            tgt_points = self.video.cotracker_track[:, tstamp].to('cuda')
            grid = get_grid(128, 128).to("cuda")

            est_flow, _, _ = interpolate(src_points=src_points, tgt_points=tgt_points, grid=grid, version="torch3d")

            # check motion magnitude / add new frame to video
            if est_flow.norm(dim=-1).mean().item() > self.thresh:
                self.count = 0
                # net, inp = self.__context_encoder(inputs[:,[0]])
                # self.net, self.inp, self.fmap = net, inp, gmap
                self.video.append(tstamp, None, None, depth, intrinsics / 4.0, image_dot)
                # self.video.append(tstamp, image[0], None, None, depth, intrinsics / 4.0, gmap, net[0], inp[0], image_dot)
                self.last_tstamp = tstamp
            else:
                self.video.append(image_dot)
                self.count += 1

        torch.cuda.empty_cache()

    def get_rainbow_colors(self, size):
        col_map = colormaps["jet"]
        col_range = np.array(range(size)) / (size - 1)
        col = torch.from_numpy(col_map(col_range)[..., :3]).float()
        col = col.view(-1, 3)
        return col


    def plot_traj_video(self):
        return
        print('============ generating traj video ===============')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('./traj_video.avi', fourcc, 10.0, (512, 512))
        # colors = cm.rainbow(np.linspace(0, 1, len(self.video.cotracker_track[0][-1]))) 
        colors = (self.get_rainbow_colors(64).numpy() * 255).astype('uint8')
        for idx in tqdm(range(len(self.video.cotracker_track[0])), 'visualizing track...'):
            image = cv2.UMat((self.video.image_dot[idx].squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
            points = self.video.cotracker_track[0, idx, :, :2]
            print('points.shape:', points.shape)
            # print('cotracker points:', points)
            for p_i in range(len(points)):
                # color = plt.cm.tab10(p_i)[:-1]
                color = colors[p_i % 64]
                coord = points[p_i].cpu().numpy() * 512
                # print(f'writing point {coord} to image {idx} with color {tuple(colors[p_i % 64])}')
                if coord[0] == 0 and coord[1] == 0:
                    continue
                if coord[0] < 1 or coord[0] > 511 or coord[1] < 1 or coord[1] > 511:
                    continue
                cv2.circle(image, (int(coord[0]), int(coord[1])), 4, (int(color[0]), int(color[1]), int(color[2])), -1)
            out.write(image)
        out.release()
        print('video saved')
        # assert False


# class MotionFilter:
#     """ This class is used to filter incoming frames and extract features """

#     def __init__(self, net, video, thresh=2.5, device="cuda:0"):
        
#         # split net modules
#         self.cnet = net.cnet
#         self.fnet = net.fnet
#         self.update = net.update

#         self.video = video
#         self.thresh = thresh
#         self.device = device

#         self.count = 0

#         # mean, std for image normalization
#         self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
#         self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
#     @torch.cuda.amp.autocast(enabled=True)
#     def __context_encoder(self, image):
#         """ context features """
#         x = self.cnet(image)
#         net, inp = self.cnet(image).split([128,128], dim=2)
#         return net.tanh().squeeze(0), inp.relu().squeeze(0)

#     @torch.cuda.amp.autocast(enabled=True)
#     def __feature_encoder(self, image):
#         """ features for correlation volume """
#         return self.fnet(image).squeeze(0)

#     @torch.cuda.amp.autocast(enabled=True)
#     @torch.no_grad()
#     def track(self, tstamp, image, depth=None, intrinsics=None):
#         """ main update operation - run on every frame in video """

#         Id = lietorch.SE3.Identity(1,).data.squeeze()
#         ht = image.shape[-2] // 8
#         wd = image.shape[-1] // 8

#         # normalize images
#         inputs = image[None, None, [2,1,0]].to(self.device) / 255.0
#         inputs = inputs.sub_(self.MEAN).div_(self.STDV)

#         # extract features
#         gmap = self.__feature_encoder(inputs)

#         ### always add first frame to the depth video ###
#         if self.video.counter.value == 0:
#             net, inp = self.__context_encoder(inputs)
#             self.net, self.inp, self.fmap = net, inp, gmap
#             self.video.append(tstamp, image, Id, 1.0, intrinsics / 8.0, gmap[0], net[0], inp[0])

#         ### only add new frame if there is enough motion ###
#         else:                
#             # index correlation volume
#             coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
#             corr = CorrBlock(self.fmap[None], gmap[None])(coords0)

#             # approximate flow magnitude using 1 update iteration
#             _, delta, weight = self.update(self.net[None], self.inp[None], corr)

#             # check motion magnitue / add new frame to video
#             if delta.norm(dim=-1).mean().item() > self.thresh:
#                 self.count = 0
#                 net, inp = self.__context_encoder(inputs)
#                 self.net, self.inp, self.fmap = net, inp, gmap
#                 self.video.append(tstamp, image, None, None, intrinsics / 8.0, gmap[0], net[0], inp[0])

#             else:
#                 self.count += 1
