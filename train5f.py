import model5f as model
import torch as t
from torch import nn
from torch.utils.data import Dataset, DataLoader
import loader2 as lo
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
def MSELoss1(g_out,fut):
    x_p=g_out[:,:,0]
    y_p = g_out[:, :, 1]
    x = fut[:, :, 0]
    y = fut[:, :, 1]
    acc=t.pow(x_p-x, 2) + t.pow(y_p-y, 2)
    out=t.sum(acc)/(128*25)
    return out

def MSELoss2(g_out,fut,mask):
    acc = t.zeros_like(mask)
    muX = g_out[:, :, 0]
    muY = g_out[:, :, 1]
    x = fut[:, :, 0]
    y = fut[:, :, 1]
    out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = t.sum(acc) / t.sum(mask)
    return lossVal

def main():
    learning_rate=0.0005
    BCEloss=nn.BCELoss().cuda()
    MSEloss=nn.MSELoss().cuda()




    generator=model.Generator()
    #generator.load_state_dict(t.load('checkpoints/4fx/epoch0_g.tar'))
    discriminator=model.Discriminator()
    #discriminator.load_state_dict(t.load('checkpoints/4fx/epoch0_d.tar'))
    classified=model.Classified()
    #classified.load_state_dict(t.load('checkpoints/4fx/epoch0_c.tar'))
    generator=generator.cuda()
    discriminator=discriminator.cuda()
    classified=classified.cuda()

    generator.train()
    discriminator.train()
    classified.train()
    t1 = lo.NgsimDataset('data/5feature/TrainSet.mat')
    t2= lo.NgsimDataset('data/5feature/ValSet.mat')
    trainDataloader = DataLoader(t1, batch_size=128, shuffle=True, num_workers=8, collate_fn=t1.collate_fn)#46272batch
    valDataloader=DataLoader(t2, batch_size=128, shuffle=True, num_workers=8, collate_fn=t2.collate_fn)#6716batch
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)
    optimizer_c = optim.Adam(classified.parameters(), lr=learning_rate)

    scheduler_g = ExponentialLR(optimizer_g, gamma=0.6)
    scheduler_d = ExponentialLR(optimizer_d, gamma=0.5)
    scheduler_c = ExponentialLR(optimizer_c, gamma=0.6)
    file=open('./checkpoints/6f/loss.txt','w')

    for epoch in range(6):
        print("epoch:", epoch, 'lr', optimizer_d.param_groups[0]['lr'])
        loss_gi1=0
        loss_gix=0
        loss_gi3 = 0
        loss_gi4 = 0
        for idx, data in enumerate(trainDataloader):
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask,va,nbrsva,lane,nbrslane,dis,nbrsdis = data
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            lat_enc=lat_enc.cuda()
            lon_enc=lon_enc.cuda()
            fut=fut.cuda()
            op_mask=op_mask.cuda()
            va = va.cuda()
            nbrsva = nbrsva.cuda()
            lane=lane.cuda()
            nbrslane=nbrslane.cuda()
            dis = dis.cuda()
            nbrsdis=nbrsdis.cuda()

            #C训练
            traj=t.cat((hist,fut),0)
            c_out=classified(traj)
            loss_c=BCEloss(c_out,lat_enc)
            optimizer_c.zero_grad()
            loss_c.backward()
            a = t.nn.utils.clip_grad_norm_(classified.parameters(), 10)
            optimizer_c.step()
            #D训练
            real_data,_=discriminator(traj, nbrs, mask)
            g_out,_,_=generator(hist, nbrs, mask, lat_enc, lon_enc,va,nbrsva,lane,nbrslane,dis,nbrsdis)
            fake_data,_=discriminator(t.cat((hist,g_out),0), nbrs, mask)
            real_label=t.ones_like(real_data)
            fake_label=t.zeros_like( fake_data)
            loss_d1=BCEloss(real_data,real_label)
            loss_d2=BCEloss(fake_data,fake_label)
            loss_d=loss_d1+loss_d2
            optimizer_d.zero_grad()
            loss_d.backward()
            a = t.nn.utils.clip_grad_norm_(discriminator.parameters(), 10)
            optimizer_d.step()
            #G训练
            g_out,lat_pred, lon_pred=generator(hist, nbrs, mask, lat_enc, lon_enc,va,nbrsva,lane,nbrslane,dis,nbrsdis)
            loss_g1 = MSELoss2(g_out, fut,op_mask)
            loss_gx=BCEloss(lat_pred,lat_enc)+BCEloss(lon_pred,lon_enc)
            traj_fake=t.cat((hist,g_out),0)
            traj_true=t.cat((hist,fut),0)
            c_out=classified(traj_fake)
            loss_g3=BCEloss(c_out,lat_enc)
            _,outp1=discriminator(traj_fake, nbrs, mask)
            _,outp2=discriminator(traj_true, nbrs, mask)
            loss_g4=MSEloss(outp1,outp2)
            loss_g=loss_g1+loss_gx+5*loss_g3+5*loss_g4
            optimizer_g.zero_grad()
            loss_g.backward()
            a = t.nn.utils.clip_grad_norm_(generator.parameters(), 10)
            optimizer_g.step()
            loss_gi1+=loss_g1.item()
            loss_gix+=loss_gx.item()
            loss_gi3 += loss_g3.item()
            loss_gi4 += loss_g4.item()
            if idx%100==99:
                print('mse:',loss_gi1/100,'|c1:',loss_gix/100,'|c:',loss_gi3/100,'|d:',loss_gi4/100)
                file.write(str(loss_gi1/100) + ',')
                loss_gi1=0
                loss_gix = 0
                loss_gi3 = 0
                loss_gi4 = 0

        avg_val_loss=0
        val_batch_count=0
        print('startval:')
        for i, data  in enumerate(valDataloader):
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask,va,nbrsva,lane,nbrslane,dis,nbrsdis = data

            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()
            va=va.cuda()
            nbrsva=nbrsva.cuda()
            lane = lane.cuda()
            nbrslane = nbrslane.cuda()
            dis=dis.cuda()
            nbrsdis=nbrsdis.cuda()

            fut_pred,_,_ = generator(hist, nbrs, mask, lat_enc, lon_enc,va,nbrsva,lane,nbrslane,dis,nbrsdis)
            l = MSELoss2(fut_pred, fut, op_mask)

            avg_val_loss += l.item()
            val_batch_count += 1

        print('valmse:',avg_val_loss/val_batch_count)
        t.save(generator.state_dict(), 'checkpoints/6f/epoch'+str(epoch+1)+'_g.tar')
        t.save(discriminator.state_dict(), 'checkpoints/6f/epoch'+str(epoch+1)+'_d.tar')
        t.save(classified.state_dict(), 'checkpoints/6f/epoch'+str(epoch+1)+'_c.tar')
        scheduler_g.step()
        scheduler_d.step()
        scheduler_c.step()
    file.close()

if __name__=='__main__':
    main()