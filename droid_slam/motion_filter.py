import cv2
import torch
import lietorch

from collections import OrderedDict
from droid_net import DroidNet

import geom.projective_ops as pops
from modules.corr import CorrBlock
import torch.nn.functional as F

from thirdparty.DOT.dot.models.point_tracking import PointTracker


class MotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net, video, thresh=2.5, device="cuda:0",
                    tracker_config="thirdparty/DOT/configs/cotracker2_patch_4_wind_8.json",
                    tracker_path="thirdparty/DOT/checkpoints/movi_f_cotracker2_patch_4_wind_8.pth",
                    estimator_config="thirdparty/DOT/configs/raft_patch_8.json",
                    estimator_path="thirdparty/DOT/checkpoints/cvo_raft_patch_8.pth"):
        
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        tracker_height, tracker_width = 512, 512
        self.online_point_tracker = PointTracker(tracker_height, tracker_width, tracker_config, tracker_path, estimator_config, estimator_path, isOnline=True).to('cuda')
        self.buffer = []
        self.is_first_step = True

    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)


    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track_buffer(self, tstamp, image, depth=None, intrinsics=None, image_dot=None):
        self.buffer.append(image_dot.to('cuda'))
        target_batch_size=4 #window
        self.track(tstamp, image, depth=depth, intrinsics=intrinsics, image_dot=image_dot) # add images to video (both the org and the reshaped one)
        if len(self.buffer)%target_batch_size==0 and len(self.buffer)!=0:

            data = {}
            data["video_chunk"] = torch.stack(self.buffer[-4*2:], dim=1).permute(1, 0, 2, 3)[None]   # video =(Batch, frames, channel, height, width)
            print('track_buffer: data["video_chunk"].shape', data["video_chunk"].shape)
            B, T, C, h, w = data["video_chunk"].shape

            H, W = 512,512 #self.resolution
            if h != H or w != W: #Reshape the frames to RAFT input size (512x512)
                data["video_chunk"] = data["video_chunk"].reshape(B * T, C, h, w)
                data["video_chunk"] = F.interpolate(data["video_chunk"], size=(H, W), mode="bilinear")
                data["video_chunk"] = data["video_chunk"].reshape(B, T, C, H, W)
            self.video.cotracker_track = self.online_point_tracker(data, mode="tracks_at_motion_boundaries_online_droid")["tracks"]
            if self.is_first_step:
                self.is_first_step = False
            else:
                self.video.cotracker_track = torch.stack([self.video.cotracker_track[..., 0] / (w - 1), self.video.cotracker_track[..., 1] / (h - 1), self.video.cotracker_track[..., 2]], dim=-1)
                print("track : self.video.cotracker_track.shape", self.video.cotracker_track.shape)
                # print("track : self.video.cotracker_track", self.video.cotracker_track)

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None, image_dot=None):
        """ main update operation - run on every frame in video """

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        # normalize images
        inputs = image[None, :, [2,1,0]].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)

        
        #### TODO: Here we could add the cotracker online function (every frame)
        ## self.video.cotracker(image) ## add the new image to cotracker
        #### TODO: (for future!) give queries based on Harris Corner Detecor (according to Tobias) or other features 

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            self.video.append(tstamp, image[0], Id, 1.0, depth, intrinsics / 8.0, gmap, net[0,0], inp[0,0], image_dot)

        ### only add new frame if there is enough motion ###
        else:                
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)

            # check motion magnitue / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh:
                self.count = 0
                net, inp = self.__context_encoder(inputs[:,[0]])
                self.net, self.inp, self.fmap = net, inp, gmap
                self.video.append(tstamp, image[0], None, None, depth, intrinsics / 8.0, gmap, net[0], inp[0], image_dot)
            else:
                self.video.append(image_dot)
                self.count += 1




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

