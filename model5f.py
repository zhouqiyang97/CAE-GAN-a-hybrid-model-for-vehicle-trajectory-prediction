import torch as t
from torch import nn
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(2, 32)
        self.lstm = nn.LSTM(32, 64)
        self.linear2 = nn.Linear(64, 32)
        self.relu = nn.LeakyReLU(0.1)
    def forward(self,hist):
        hist_enc = self.relu(self.linear1(hist))
        _, (hist_enc, _) = self.lstm(hist_enc)
        hist_enc = hist_enc.view(hist_enc.size(1), hist_enc.size(2))
        hist_enc = self.relu(self.linear2(hist_enc))
        return hist_enc
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.linear1=nn.Linear(2,32)
        self.lstm=nn.LSTM(32,64)
        self.linear2=nn.Linear(64,32)
        self.relu=nn.LeakyReLU(0.1)
        self.soc_conv = t.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv_3x1 = t.nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3, 1))
        self.soc_maxpool = t.nn.MaxPool2d((2, 1), padding=(1, 0))
    def forward(self,hist,nbrs,mask):   #[len1,batch,2]
        hist_enc=self.relu(self.linear1(hist))
        _,(hist_enc,_)=self.lstm(hist_enc)
        hist_enc=hist_enc.view(hist_enc.size(1),hist_enc.size(2))
        hist_enc=self.relu(self.linear2(hist_enc))

        nbrs_enc=self.relu(self.linear1(nbrs))
        _,(nbrs_enc,_)=self.lstm(nbrs_enc)
        nbrs_enc=nbrs_enc.view(nbrs_enc.size(1),nbrs_enc.size(2))
        soc_enc = t.zeros_like(mask).float()
        soc_enc = soc_enc.masked_scatter_(mask, nbrs_enc)
        soc_enc = soc_enc.permute(0,3,2,1)         #(batch,input_l,13,3)
        soc_enc=self.relu(self.soc_conv(soc_enc))
        soc_enc = self.soc_maxpool(self.relu(self.conv_3x1(soc_enc)))
        soc_enc =soc_enc.view(-1,5*32)
        enc = t.cat((soc_enc, hist_enc), 1)
        return enc#[batch,112]

class VAE5f(nn.Module):
    def __init__(self):
        super(VAE5f, self).__init__()
        self.linear1=nn.Linear(11,32)
        self.lstm=nn.LSTM(32,64)
        self.linear2=nn.Linear(64,32)
        self.relu=nn.LeakyReLU(0.1)
        self.soc_conv = t.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv_3x1 = t.nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3, 1))
        self.soc_maxpool = t.nn.MaxPool2d((2, 1), padding=(1, 0))
    def forward(self,hist,nbrs,mask,va,nbrsva,lane,nbrslane):   #[len1,batch,2]
        hist5f=t.cat((hist,va,lane),2)
        nbrs5f=t.cat((nbrs,nbrsva,nbrslane),2)
        hist_enc=self.relu(self.linear1(hist5f))
        _,(hist_enc,_)=self.lstm(hist_enc)
        hist_enc=hist_enc.view(hist_enc.size(1),hist_enc.size(2))
        hist_enc=self.relu(self.linear2(hist_enc))

        nbrs_enc=self.relu(self.linear1(nbrs5f))
        _,(nbrs_enc,_)=self.lstm(nbrs_enc)
        nbrs_enc=nbrs_enc.view(nbrs_enc.size(1),nbrs_enc.size(2))
        soc_enc = t.zeros_like(mask).float()
        soc_enc = soc_enc.masked_scatter_(mask, nbrs_enc)
        soc_enc = soc_enc.permute(0,3,2,1)         #(batch,input_l,13,3)
        soc_enc=self.relu(self.soc_conv(soc_enc))
        soc_enc = self.soc_maxpool(self.relu(self.conv_3x1(soc_enc)))
        soc_enc =soc_enc.view(-1,5*32)
        enc = t.cat((soc_enc, hist_enc), 1)
        return enc#[batch,112]


class Decoder(nn.Module):
    def __init__(self,out_length=25):
        super(Decoder,self).__init__()
        self.out_length=out_length
        self.lstm=t.nn.LSTM(197, 128)
        self.linear1=nn.Linear(128,2)
    def forward(self,feature):
        feature = feature.repeat(self.out_length, 1, 1)
        h_dec, _ = self.lstm(feature)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.linear1(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        return fut_pred

class Generator(nn.Module):
    def __init__(self,train_flag=True):
        super(Generator, self).__init__()
        self.VAE5f=VAE5f()
        self.Decoder=Decoder()
        self.lat_linear = nn.Linear(192, 3)
        self.lon_linear = nn.Linear(192, 2)
        self.train_flag=train_flag
        self.softmax=nn.Softmax(dim=1)
    def forward(self,hist, nbrs, mask, lat_enc, lon_enc,va,nbrsva,lane,nbrslane):
        enc=self.VAE5f(hist, nbrs, mask,va,nbrsva,lane,nbrslane)
        lat_pred=self.softmax(self.lat_linear(enc))
        lon_pred=self.softmax(self.lon_linear(enc))
        if self.train_flag:
            out=t.cat((enc, lat_enc, lon_enc),1)   #(128,197)
            out=self.Decoder(out)
            return out,lat_pred,lon_pred
        else:
            out=[]
            for k in range(2):
                for l in range(3):
                    lat_enc_tmp = t.zeros_like(lat_enc)
                    lon_enc_tmp = t.zeros_like(lon_enc)
                    lat_enc_tmp[:, l] = 1
                    lon_enc_tmp[:, k] = 1
                    enc_tmp = t.cat((enc, lat_enc_tmp, lon_enc_tmp), 1)
                    out.append(self.Decoder(enc_tmp))
            return out, lat_pred, lon_pred

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.VAE=VAE()
        self.mlp1=nn.Sequential(nn.Linear(192,1024),
                               nn.LeakyReLU(0.1),
                               nn.Linear(1024,1024),
                               nn.LeakyReLU(0.1))
        self.mlp2=nn.Sequential(nn.Linear(1024, 1),
                               nn.Sigmoid())
    def forward(self,traj,nbrs, mask):
        out=self.VAE(traj,nbrs, mask)
        outp=self.mlp1(out)
        out=self.mlp2(outp)
        return out,outp

class Classified(nn.Module):
    def __init__(self):
        super(Classified, self).__init__()
        self.Encoder=Encoder()
        self.mlp=nn.Sequential(nn.Linear(32,1024),
                               nn.LeakyReLU(0.1),
                               nn.Linear(1024,1024),
                               nn.LeakyReLU(0.1),
                               nn.Linear(1024, 3),
                               nn.Softmax(dim=1))
    def forward(self,traj):
        out=self.Encoder(traj)
        out=self.mlp(out)
        return out