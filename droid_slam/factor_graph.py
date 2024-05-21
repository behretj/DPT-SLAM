import torch
import lietorch
import numpy as np
import sys
import os

import matplotlib.pyplot as plt
from lietorch import SE3
from modules.corr import CorrBlock, AltCorrBlock
import geom.projective_ops as pops

from thirdparty.DOT.dot.models.optical_flow import OpticalFlow

class FactorGraph:
    # def __init__(self, video, update_op, device="cuda:0", corr_impl="volume", max_factors=-1, upsample=False):
    def __init__(self, video, device="cuda:0", corr_impl="volume", max_factors=-1, upsample=False):
        self.video = video
        # self.update_op = update_op
        self.device = device
        self.max_factors = max_factors
        # self.corr_impl = corr_impl
        self.upsample = upsample

        # operator at 1/4 resolution
        video.ht = 128
        video.wd = 128
        self.ht = ht = 128
        self.wd = wd = 128

        self.tracks_thresh = 512

        self.coords0 = pops.coords_grid(ht, wd, device=device)
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        # self.corr, self.net, self.inp = None, None, None
        self.damping = 1e-6 * torch.ones_like(self.video.disps)

        # inactive factors
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.ii_bad = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_bad = torch.as_tensor([], dtype=torch.long, device=device)

        # self.video.optical_flow_refiner = OpticalFlow(height=512, width=512,
        #                                         config='./thirdparty/DOT/dot/configs/raft_patch_4_alpha.json',
        #                                         load_path='./thirdparty/DOT/dot/checkpoints/movi_f_raft_patch_4_alpha.pth').cuda()


    def __filter_repeated_edges(self, ii, jj):
        """ remove duplicate edges """

        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)] +
            [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)])

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    def print_edges(self):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        w = torch.mean(self.weight, dim=[0,2,3,4]).cpu().numpy()
        w = w[ix]
        for e in zip(ii, jj, w):
            print(e)
        print()

    def filter_edges(self):
        """ remove bad edges """
        conf = torch.mean(self.weight, dim=[0,2,3,4])
        mask = (torch.abs(self.ii-self.jj) > 2) & (conf < 0.001)

        self.ii_bad = torch.cat([self.ii_bad, self.ii[mask]])
        self.jj_bad = torch.cat([self.jj_bad, self.jj[mask]])
        self.rm_factors(mask, store=False)

    def clear_edges(self):
        self.rm_factors(self.ii >= 0)
        # self.net = None
        # self.inp = None

    @torch.cuda.amp.autocast(enabled=True)
    def add_factors(self, ii, jj, remove=False):
        """ add edges to factor graph """
        # print(f'add_factores ii: {ii}, jj: {jj}')

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges
        ii, jj = self.__filter_repeated_edges(ii, jj)

        if ii.shape[0] == 0:
            return

        # place limit on number of factors
        if self.max_factors > 0 and self.ii.shape[0] + ii.shape[0] > self.max_factors \
                and remove:
                # and self.corr is not None and remove:
            
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        # net = self.video.nets[ii].to(self.device).unsqueeze(0)

        # correlation volume for new edges
        # if self.corr_impl == "volume":
        #     c = (ii == jj).long()
        #     fmap1 = self.video.fmaps[ii,0].to(self.device).unsqueeze(0)
        #     fmap2 = self.video.fmaps[jj,c].to(self.device).unsqueeze(0)
        #     corr = CorrBlock(fmap1, fmap2)
        #     self.corr = corr if self.corr is None else self.corr.cat(corr)

        #     inp = self.video.inps[ii].to(self.device).unsqueeze(0)
        #     self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        with torch.cuda.amp.autocast(enabled=False):
            track = self.video.cotracker_track
            
            tstamps_list = (self.video.tstamp).tolist()
            # print('factor_graph, add_factor: tstamps_list', tstamps_list)
            # print(f'-------- add_factores ii: {ii}, jj: {jj}')
            ii_list = (ii).tolist()
            jj_list = (jj).tolist()

            sources_list = [int(tstamps_list[i]) for i in ii_list]
            targets_list = [int(tstamps_list[i]) for i in jj_list]
            # print('sources_list', sources_list)
            # print('targets_list', targets_list)
            video = self.video.image_dot

            self.video.optical_flow_refiner(track, mode="flow_between_frames", video=video, ii=sources_list, jj=targets_list)
            torch.cuda.empty_cache()

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        # reprojection factors
        # self.net = net if self.net is None else torch.cat([self.net, net], 1)


    @torch.cuda.amp.autocast(enabled=True)
    def rm_factors(self, mask, store=False):
        """ drop edges from factor graph """

        tstamps_list = (self.video.tstamp).tolist()
        ii_list = (self.ii[mask]).tolist()
        jj_list = (self.jj[mask]).tolist()

        sources_list = [int(tstamps_list[i]) for i in ii_list]
        targets_list = [int(tstamps_list[i]) for i in jj_list]

        # store estimated factors
        if store:
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            # self.target_inac = torch.cat([self.target_inac, self.target[:,mask]], 1)
            # self.weight_inac = torch.cat([self.weight_inac, self.weight[:,mask]], 1)

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]
        
        # if self.corr_impl == "volume":
        #     self.corr = self.corr[~mask]

        # if self.net is not None:
        #     self.net = self.net[:,~mask]

        # if self.inp is not None:
        #     self.inp = self.inp[:,~mask]

        self.video.optical_flow_refiner.rm_flows(sources_list, targets_list, store=store)

        # self.target = self.target[:,~mask]
        # self.weight = self.weight[:,~mask]

        torch.cuda.empty_cache()




    @torch.cuda.amp.autocast(enabled=True)
    def rm_keyframe(self, ix):
        """ drop edges from factor graph """

        with self.video.get_lock():
            # self.video.images[ix] = self.video.images[ix+1]
            self.video.poses[ix] = self.video.poses[ix+1]
            self.video.disps[ix] = self.video.disps[ix+1]
            self.video.disps_sens[ix] = self.video.disps_sens[ix+1]
            self.video.intrinsics[ix] = self.video.intrinsics[ix+1]

            # self.video.nets[ix] = self.video.nets[ix+1]
            # self.video.inps[ix] = self.video.inps[ix+1]
            # self.video.fmaps[ix] = self.video.fmaps[ix+1]

        m = (self.ii_inac == ix) | (self.jj_inac == ix)

        if torch.any(m):

            tstamps_list = (self.video.tstamp).tolist()
            ii_list = (self.ii_inac[m]).tolist()
            jj_list = (self.jj_inac[m]).tolist()

            sources_list = [int(tstamps_list[i]) for i in ii_list]
            targets_list = [int(tstamps_list[i]) for i in jj_list]
            self.video.optical_flow_refiner.reset_inac(sources_list, targets_list)

            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]

            # self.target_inac = self.target_inac[:,~m]
            # self.weight_inac = self.weight_inac[:,~m]

        m = (self.ii == ix) | (self.jj == ix)

        self.rm_factors(m, store=False)

        self.ii_inac[self.ii_inac > ix] -= 1
        self.jj_inac[self.jj_inac > ix] -= 1
        
        self.ii[self.ii > ix] -= 1
        self.jj[self.jj > ix] -= 1

        # update tstamp removing frame ix
        with self.video.get_lock():
            self.video.tstamp[ix:-1] = self.video.tstamp[ix+1:].clone()
            self.video.tstamp[-1] = 0.0


    def add_neighborhood_factors(self, t0, t1, r=3):
        """ add edges between neighboring frames within radius r """

        ii, jj = torch.meshgrid(torch.arange(t0,t1), torch.arange(t0,t1))
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        c = 1 if self.video.stereo else 0

        keep = ((ii - jj).abs() > c) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])

    
    def add_proximity_factors(self, t0=0, t1=0, rad=2, nms=2, beta=0.25, thresh=16.0, remove=False, init=False):
        """ add edges to the factor graph based on distance """

        t = self.video.counter.value
        ix = torch.arange(t0, t)
        jx = torch.arange(t1, t)

        ii, jj = torch.meshgrid(ix, jx)
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        d, num_tracks_used = self.video.distance(ii, jj, beta=beta)

        d[ii - rad < jj] = np.inf
        d[d > 100] = np.inf
        # don't do pairs if flow between the two frames is obtained with not enough tracks
        # TODO: maybe not be useful the init part, just do it always
        if not init:
            d[num_tracks_used < self.tracks_thresh] = np.inf

        ii1 = torch.cat([self.ii, self.ii_bad, self.ii_inac], 0)
        jj1 = torch.cat([self.jj, self.jj_bad, self.jj_inac], 0)
        for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf


        es = []
        for i in range(t0, t):
            if self.video.stereo:
                es.append((i, i))
                d[(i-t0)*(t-t1) + (i-t1)] = np.inf

            for j in range(max(i-rad-1,0), i):
                es.append((i,j))
                es.append((j,i))
                d[(i-t0)*(t-t1) + (j-t1)] = np.inf

        ix = torch.argsort(d)
        for k in ix:
            if d[k].item() > thresh:
                continue

            if len(es) > self.max_factors:
                break

            i = ii[k]
            j = jj[k]
            
            # bidirectional
            es.append((i, j))
            es.append((j, i))

            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf

        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove)


    """
    Update function which performs BA
    - substitutes update of DROID
    """
    # TODO: check if autocast is necessary if we only use BA and not droid's GRU
    @torch.cuda.amp.autocast(enabled=True)
    def update_DOT_SLAM(self, t0=None, t1=None, itrs=2, ba_calls=4, use_inactive=False, EP=1e-7, motion_only=False):
        """ run update operator on factor graph """

        if t0 is None:
            t0 = max(1, self.ii.min().item()+1)

        with torch.cuda.amp.autocast(enabled=False):

            if use_inactive:
                m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3)
                num_inac = len(self.ii_inac[m])
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)

            else:
                ii, jj = self.ii, self.jj

            tstamps_list = (self.video.tstamp).tolist()
            ii_list = (ii).tolist()
            jj_list = (jj).tolist()

            sources_list = [int(tstamps_list[i]) for i in ii_list]
            targets_list = [int(tstamps_list[i]) for i in jj_list]

            # Iterate over the ii and jj lists
            l = len(sources_list)
            target, weight = [], []
            for idx in range(l):
                i = sources_list[idx]
                j = targets_list[idx]
                # Get the flow and weight from the dictionaries
                if use_inactive:
                    if idx < num_inac:
                        flow = self.coords0 + self.video.optical_flow_refiner.refined_flow_inac[i][j]#.to('cuda')
                        w = self.video.optical_flow_refiner.refined_weight_inac[i][j]#.to('cuda')
                    else:
                        flow = self.coords0 + self.video.optical_flow_refiner.refined_flow[i][j]#.to('cuda')
                        w = self.video.optical_flow_refiner.refined_weight[i][j]#.to('cuda')
                else:
                    flow = self.coords0 + self.video.optical_flow_refiner.refined_flow[i][j]#.to('cuda')
                    w = self.video.optical_flow_refiner.refined_weight[i][j]#.to('cuda')

                target.append(flow)
                weight.append(w)

            # Convert the lists to PyTorch tensors and add the necessary dimensions
            target = torch.stack(target, dim=0).to(device="cuda", dtype=torch.float)[None]
            weight = torch.stack(weight, dim=0).to(device="cuda", dtype=torch.float)[None]

            target = target.squeeze(0).permute(0, 3, 1, 2).contiguous()
            weight = weight.squeeze(0).permute(0, 3, 1, 2).contiguous()

            # Input fixed damping values (eta)
            damping = torch.full((torch.unique(ii).size(0), target.shape[2], target.shape[3]),  0.005).to(device="cuda", dtype=torch.float).contiguous()

            for iters in range(ba_calls):
                # after 2 BA calls with higher damping use lower damping for successive iterations
                if (iters == 1):
                    damping = torch.full((torch.unique(ii).size(0), target.shape[2], target.shape[3]),  0.0001).to(device="cuda", dtype=torch.float).contiguous()
                elif (iters == 2):
                    damping = torch.full((torch.unique(ii).size(0), target.shape[2], target.shape[3]),  1e-6).to(device="cuda", dtype=torch.float).contiguous()
                
                # dense bundle adjustment
                self.video.ba(target, weight, damping, ii, jj, t0, t1, 
                    itrs=itrs, lm=1e-4, ep=0.1, motion_only=motion_only)
        
            # WE DON'T HAVE THE UPMASK ANYMORE IF WE DON'T USE DROID GRU
            # if self.upsample:
            #     self.video.upsample(torch.unique(self.ii), upmask)

        self.age += 1