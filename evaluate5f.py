from __future__ import print_function
import torch as t
import model5f as model
import loader2 as lo
from torch.utils.data import DataLoader

def maskedMSETest(y_pred, y_gt, mask):
    acc = t.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = t.sum(acc[:,:,0],dim=1)
    counts = t.sum(mask[:,:,0],dim=1)
    return lossVal, counts





def main(epoch_last):
    # Initialize network
    generator = model.Generator(train_flag=True)
    generator.load_state_dict(t.load('checkpoints/5f/huatu/epoch'+str(epoch_last)+'_g.tar'))
    generator = generator.cuda()

    t2= lo.NgsimDataset('data/5feature/ValSet.mat')
    valDataloader=DataLoader(t2, batch_size=128, shuffle=True, num_workers=8, collate_fn=t2.collate_fn)

    lossVals = t.zeros(25).cuda()
    counts = t.zeros(25).cuda()

    for idx, data in enumerate(valDataloader):
        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, va, nbrsva, lane, nbrslane ,_,_= data


        hist = hist.cuda()
        nbrs = nbrs.cuda()
        mask = mask.cuda()
        lat_enc = lat_enc.cuda()
        lon_enc = lon_enc.cuda()
        fut = fut.cuda()
        op_mask = op_mask.cuda()
        va = va.cuda()
        nbrsva = nbrsva.cuda()
        lane = lane.cuda()
        nbrslane = nbrslane.cuda()

        fut_pred, lat_pred, lon_pred = generator(hist, nbrs, mask, lat_enc, lon_enc,va,nbrsva,lane,nbrslane)
        # fut_pred_max = t.zeros_like(fut_pred[0])
        # for k in range(lat_pred.shape[0]):  # 128
        #     lat_man = t.argmax(lat_pred[k, :]).detach()
        #     lon_man = t.argmax(lon_pred[k, :]).detach()
        #     index = lon_man * 3 + lat_man
        #     fut_pred_max[:, k, :] = fut_pred[index][:, k, :]  # (128,5)/
        # for lat_enc_i in range(lat_enc.size(0)):
        #     if lat_enc[lat_enc_i,0]==1:
        #         l=t.pow(fut_pred[:,lat_enc_i,0]-fut[:,lat_enc_i,0],2)+t.pow(fut_pred[:,lat_enc_i,1]-fut[:,lat_enc_i,1],2)
        #         l=t.pow(l,0.5)
        #         c=t.ones(25).cuda()
        #         lossVals+=l.detach()
        #         counts += c.detach()
        #             #print(lossVals,counts)

        l, c = maskedMSETest(fut_pred, fut, op_mask)
        lossVals += l.detach()
        counts += c.detach()

    print(t.pow(lossVals / counts, 0.5) * 0.3048)  # Calculate RMSE and convert from feet to meters
    #print(lossVals/counts*0.3048)
if __name__ == '__main__':
    #for epoch in range(20):
    main(3)