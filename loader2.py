from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch
class NgsimDataset(Dataset):


    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size = 64, grid_size = (13,3)):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = t_h  # 历史轨迹长度
        self.t_f = t_f  # 预测轨迹长度
        self.d_s = d_s  # skip
        self.enc_size = enc_size # size of encoder LSTM
        self.grid_size = grid_size # size of social context grid



    def __len__(self):
        return len(self.D)



    def __getitem__(self, idx):

        dsId = self.D[idx, 0].astype(int)   #dataset id
        vehId = self.D[idx, 1].astype(int)  #车辆id
        t = self.D[idx, 2]                  #帧数
        grid = self.D[idx,10:]               #各个格子的车的id
        neighbors = []
        neighborsva=[]
        neighborslane=[]
        neighborsdistance = []


        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId,t,vehId,dsId)      #返回自车的历史相对轨迹（t_h,2）,如果不满足长度，返回empty
        refdistance = np.zeros_like(hist[:,0])
        refdistance=refdistance.reshape(len(refdistance),1)
        fut = self.getFuture(vehId,t,dsId)              #返回自车的未来相对轨迹（t_h,2）
        va=self.getVA(vehId,t,vehId,dsId)
        lanexx=self.getLane(vehId,t,vehId,dsId)
        lane=[int(i) for i in lanexx]



        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            nbrsdis=self.getHistory(i.astype(int), t,vehId,dsId)
            if nbrsdis.shape!=(0,2):
                uu = np.power(hist - nbrsdis, 2)
                distancexxx = np.sqrt(uu[:, 0] + uu[:, 1])
                distancexxx = distancexxx.reshape(len(distancexxx), 1)
            else:
                distancexxx=np.empty([0,1])
            neighbors.append(nbrsdis)  #返回旁车序列的list[array,array,....]
            neighborsva.append(self.getVA(i.astype(int),t,vehId,dsId))
            neighborslane.append(self.getLane(i.astype(int),t,vehId,dsId))
            neighborsdistance.append(distancexxx)
        lon_enc = np.zeros([2])
        lon_enc[int(self.D[idx, 9] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 8] - 1)] = 1


        #hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

        return hist,fut,neighbors,lat_enc,lon_enc,va,neighborsva,lane,neighborslane,refdistance,neighborsdistance

    def getLane(self,vehId,t,refVehId,dsId):
        if vehId == 0:
            return np.empty([0,1])
        else:
            if self.T.shape[1]<=vehId-1:
                return np.empty([0,1])
            refTrack = self.T[dsId-1][refVehId-1].transpose()       #3*refvehid车的序列长的np（帧数，x，y,v,a）
            vehTrack = self.T[dsId-1][vehId-1].transpose()          #3*vehid车的序列长的np（帧数，x，y,v,a）
            refPos = refTrack[np.where(refTrack[:,0]==t)][0,5]    #refvehid车在第t帧的（v，a）

            if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
                 return np.empty([0,1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s,5]

            if len(hist) < self.t_h//self.d_s + 1:
                return np.empty([0,1])
            return hist


    def getVA(self,vehId,t,refVehId,dsId):
        if vehId == 0:
            return np.empty([0,2])
        else:
            if self.T.shape[1]<=vehId-1:
                return np.empty([0,2])
            refTrack = self.T[dsId-1][refVehId-1].transpose()       #3*refvehid车的序列长的np（帧数，x，y,v,a）
            vehTrack = self.T[dsId-1][vehId-1].transpose()          #3*vehid车的序列长的np（帧数，x，y,v,a）
            refPos = refTrack[np.where(refTrack[:,0]==t)][0,3:5]    #refvehid车在第t帧的（v，a）

            if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
                 return np.empty([0,2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s,3:5]-refPos

            if len(hist) < self.t_h//self.d_s + 1:
                return np.empty([0,2])
            return hist

    ## Helper function to get track history
    def getHistory(self,vehId,t,refVehId,dsId):
        if vehId == 0:
            return np.empty([0,2])
        else:
            if self.T.shape[1]<=vehId-1:
                return np.empty([0,2])
            refTrack = self.T[dsId-1][refVehId-1].transpose()       #3*refvehid车的序列长的np（帧数，x，y）
            vehTrack = self.T[dsId-1][vehId-1].transpose()          #3*vehid车的序列长的np（帧数，x，y）
            refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]    #refvehid车在第t帧的（x，y）

            if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
                 return np.empty([0,2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s,1:3]-refPos

            if len(hist) < self.t_h//self.d_s + 1:
                return np.empty([0,2])
            return hist

    def getdistance(self,vehId,t,refVehId,dsId):
        if vehId == 0:
            return np.empty([0,1])
        else:
            if self.T.shape[1]<=vehId-1:
                return np.empty([0,1])
            refTrack = self.T[dsId-1][refVehId-1].transpose()       #3*refvehid车的序列长的np（帧数，x，y）
            vehTrack = self.T[dsId-1][vehId-1].transpose()          #3*vehid车的序列长的np（帧数，x，y）
            refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]    #refvehid车在第t帧的（x，y）

            if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
                 return np.empty([0,1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
                hist_ref = refTrack[stpt:enpt:self.d_s,1:3] - refPos
                uu = np.power(hist - hist_ref, 2)
                distance = np.sqrt(uu[:, 0] + uu[:, 1])
                distance = distance.reshape(len(distance), 1)

            if len(hist) < self.t_h//self.d_s + 1:
                return np.empty([0,1])
            return distance


    ## Helper function to get track future
    def getFuture(self, vehId, t,dsId):
        vehTrack = self.T[dsId-1][vehId-1].transpose()              #3*vehid车的序列长的np（帧数，x，y）
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]    ##vehid车在第t帧的（x，y）
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
        return fut



    ## Collate function for dataloader
    def collate_fn(self, samples):

        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for _,_,nbrs,_,_,_,_,_,_,_,_ in samples:
            nbr_batch_size += sum([len(nbrs[i])!=0 for i in range(len(nbrs))])
        maxlen = self.t_h//self.d_s + 1
        nbrs_batch = torch.zeros(maxlen,nbr_batch_size,2)   #(len,batch*车数，2)
        nbrsva_batch=torch.zeros(maxlen,nbr_batch_size,2)
        nbrslane_batch=torch.zeros(maxlen,nbr_batch_size,7)
        nbrsdis_batch = torch.zeros(maxlen, nbr_batch_size, 1)


        # Initialize social mask batch:
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1],self.grid_size[0],self.enc_size) #(batch,3,13,h)
        mask_batch = mask_batch.bool()


        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen,len(samples),2)                     #(len1,batch,2)
        distance_batch = torch.zeros(maxlen, len(samples), 1)
        fut_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)          #(len2,batch,2)
        op_mask_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)      #(len2,batch,2)
        lat_enc_batch = torch.zeros(len(samples), 3)  # (batch,3)
        lon_enc_batch = torch.zeros(len(samples), 2)  # (batch,2)
        va_batch=torch.zeros(maxlen,len(samples), 2)
        lane_batch=torch.zeros(maxlen,len(samples),7)



        count = 0
        count1=0
        count2=0
        count3=0
        for sampleId,(hist, fut, nbrs,lat_enc, lon_enc,va,neighborsva,lane,neighborslane,refdistance,neighborsdistance) in enumerate(samples):

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist),sampleId,0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            distance_batch[0:len(hist), sampleId, :] = torch.from_numpy(refdistance)
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut),sampleId,:] = 1
            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            va_batch[0:len(va), sampleId, 0] = torch.from_numpy(va[:, 0])
            va_batch[0:len(va), sampleId, 1] = torch.from_numpy(va[:, 1])
            for lanet in range(len(lane)):
                lane_batch[lanet,sampleId,lane[lanet]-1]=1


            # Set up neighbor, neighbor sequence length, and mask batches:
            for id,nbr in enumerate(nbrs):
                if len(nbr)!=0:
                    nbrs_batch[0:len(nbr),count,0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId,pos[1],pos[0],:] = torch.ones(self.enc_size).byte()
                    count+=1

            for id,nbrva in enumerate(neighborsva):
                if len(nbrva)!=0:
                    nbrsva_batch[0:len(nbrva),count1,0] = torch.from_numpy(nbrva[:, 0])
                    nbrsva_batch[0:len(nbrva), count1, 1] = torch.from_numpy(nbrva[:, 1])
                    count1+=1

            for id,nbrlane in enumerate(neighborslane):
                if len(nbrlane)!=0:
                    for nbrslanet in range(len(nbrlane)):
                        nbrslane_batch[nbrslanet,count2,int(nbrlane[nbrslanet]-1)] = 1

                    count2+=1

            for id,nbrdis in enumerate(neighborsdistance):
                if len(nbrdis)!=0:
                    nbrsdis_batch[0:len(nbrdis),count3,:] = torch.from_numpy(nbrdis)
                    count3+=1

        return hist_batch,nbrs_batch, mask_batch,lat_enc_batch, lon_enc_batch, fut_batch,op_mask_batch,va_batch,nbrsva_batch,lane_batch,nbrslane_batch,distance_batch,nbrsdis_batch

