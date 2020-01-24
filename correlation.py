import torch
import numpy as np


def make_list(model):
    list=[model.module.conv1, model.module.fc, None, None]
    for k in range(len(model.module.block1.layer)):
        list.append(model.module.block1.layer[k].conv)

    for k in range(len(model.module.block2.layer)):
        list.append(model.module.block2.layer[k].conv)

    for k in range(len(model.module.block3.layer)):
        list.append(model.module.block3.layer[k].conv)
        
    try:
        list.extend([model.module.block1.layer[0].conv_res,model.module.block2.layer[0].conv_res,model.module.block3.layer[0].conv_res, None])
    except:
        misc = 0
        
    return list

def pair_correllation(m, fanin = True, fanout = True, needReshape=True, maxN = 5000):
    N= maxN
    A = m.numpy()
    
    tot = 1
    for k in range(len(A.shape)):
        tot = tot * A.shape[k]

    if needReshape:
        if fanin and fanout:
            n = A.shape[0]*A.shape[1]
        else:
            if not fanin:
                A = np.swapaxes(A, 0,1)
            n = A.shape[0]
        A = A.reshape([n,tot//n])
    else:
        n = A.shape[0]
        
    n2 = tot//n
    if n > N:

        s = 0.0
        for k in range(N):
            i = np.random.randint(0,n)
            j = np.random.randint(0,n)
            while i==j:
                i = np.random.randint(0,n)
                j = np.random.randint(0,n)

            a= A[i,:]
            b= A[j,:]

            norm1 = np.dot(a,a)
            norm2 = np.dot(b,b)
            if norm1 == 0.0 or norm2==0.0:
                s+=1.0
            else:
                s+=np.dot(a,b)/np.sqrt(norm1*norm2)
        return s/N
    else:
        ##A = A[np.random.permutation(n)[0:N],:]
        ##n = N
    ##print("n=%d, m = %d" % (n,n2) )
        B = np.corrcoef(A)
        B[np.isnan(B)]=1.0
        corr = np.sum(np.tril(B,-1))/(((n-1)*n)/2)

    return corr

def logc(x):
    return -np.log(np.abs(x))



def measure_correlation(model, epoch, N = 5000, writer = None):
    list = make_list(model)
 
    res = {} 
    for k,m in enumerate(list):
        if m is None:
            continue
        
        mm = m.weight.data
        
        if len(m.weight.data.shape)>2:

            corrs =  pair_correllation(mm.cpu(),fanin=True,fanout=True, maxN = N)
            forward = pair_correllation(mm.cpu(),fanin=False,fanout=True, maxN = N)
            backward = pair_correllation(mm.cpu(),fanin=True,fanout=False, maxN = N)

            name = "%d) %s" % (k,m)
            name = name.replace(" ",'')
            name=  name[0:31] + ')'
         
            if writer is not None:
                writer.add_scalar('%s/correlations' % name, logc(corrs), epoch)
                writer.add_scalar('%s/forward' % name, logc(forward), epoch)
                writer.add_scalar('%s/backward' % name, logc(backward), epoch)


            res["%s" % m] =  (corrs, forward, backward)
    return res
