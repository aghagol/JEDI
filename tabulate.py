import os, sys, pickle
import numpy as np
""" define some parameters """
ssss = 'simo_dirty_inpainting'
m_choices = [12, 22, 32]
""" list images """
# img_list = set([it for it in os.listdir(os.path.join('..','clean')) if it[-4:] == '.png'])
img_list = ['barbara','boat','finprint','grass','house',
'lena','man','matches','rocket','aerial','barley','bubbles']
img_lizt = ['Barbara','boat','fingerprint','grass','house',
'Lena','man','matches','shuttle','aerial','barley','bubbles']
f = open('out.txt', 'w')
for img_idx in range(len(img_list)):
    img_test = img_list[img_idx]
    img_tezt = img_lizt[img_idx]
    f.write('\hline ')
    f.write(img_tezt+' & ')
    for m in m_choices:
        psnr0 = []
        datapath = os.path.join('.',ssss,img_test,'m%d'%m)
        temp = [it for it in os.listdir(datapath) if it.endswith('.pickle')]
        for t in temp:
            with open(os.path.join(datapath,t)) as ff:
                PSNR, Verr, PSNR0, Verr0 = pickle.load(ff)
                snr0 = PSNR0.ravel()[Verr0.argmin()]
            psnr0.append(snr0)
        psnr0_avg = np.mean(psnr0)
        if m == m_choices[-1]:
            f.write('%.2f & ' %psnr0_avg)
        else:
            f.write('%.2f & ' %psnr0_avg)
    for m in m_choices:
        psnr1 = []
        datapath = os.path.join('.',ssss,img_test,'m%d'%m)
        temp = [it for it in os.listdir(datapath) if it.endswith('.pickle')]
        for t in temp:
            with open(os.path.join(datapath,t)) as ff:
                PSNR, Verr, PSNR0, Verr0 = pickle.load(ff)
                snr1 = PSNR.ravel()[Verr.argmin()]
            psnr1.append(snr1)
        psnr1_avg = np.mean(psnr1)
        if m == m_choices[-1]:
            f.write('%.2f & ' %psnr1_avg)
        else:
            f.write('%.2f & ' %psnr1_avg)
    for m in m_choices:
        psnr1 = []
        psnr0 = []
        datapath = os.path.join('.',ssss,img_test,'m%d'%m)
        temp = [it for it in os.listdir(datapath) if it.endswith('.pickle')]
        for t in temp:
            with open(os.path.join(datapath,t)) as ff:
                PSNR, Verr, PSNR0, Verr0 = pickle.load(ff)
                snr1 = PSNR.ravel()[Verr.argmin()]
                snr0 = PSNR0.ravel()[Verr0.argmin()]
            psnr1.append(snr1)
            psnr0.append(snr0)
        psnr1_avg = np.mean(psnr1)
        psnr0_avg = np.mean(psnr0)
        if m == m_choices[-1]:
            f.write('%.2f \\\\ \n' %(psnr1_avg-psnr0_avg))
        else:
            f.write('%.2f & ' %(psnr1_avg-psnr0_avg))
f.close()














