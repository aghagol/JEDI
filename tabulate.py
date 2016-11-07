import os, sys, pickle
import numpy as np
""" define some parameters """
ssss = 'simo_dirty_dct'
m_choices = [3, 6, 12, 16, 19, 25, 32]
thr = {3:3, 6:19, 12:19, 16:19, 19:19, 25:19, 32:19}
""" list images """
img_list = set([it for it in os.listdir(os.path.join('..','clean')) if it[-4:] == '.png'])
img_list = ['barbara','boat','bridge','couple','fingerprint','finprint',
'flinstones','grass','hill','house','lena','man','matches','rocket']
img_lizt = ['Barbara','boat','bridge','couple','fingerprint 1','fingerprint 2',
'Flintstones','grass','hill','house','Lena','man','matches','shuttle']
f = open('out.txt', 'w')
for img_idx in range(len(img_list)):
    img_test = img_list[img_idx]
    img_tezt = img_lizt[img_idx]
    f.write('\hline\n')
    f.write(img_tezt+' & \n')
    for m in m_choices:
        psnr0 = []
        datapath = os.path.join('.',ssss,img_test,'m%d'%m)
        temp = [it for it in os.listdir(datapath) if it[-7:] == '.pickle']
        for t in temp:
            with open(os.path.join(datapath,t)) as ff:
                snr1, snr0 = pickle.load(ff)
            psnr0.append(snr0)
        psnr0_avg = np.mean(psnr0)
        if m == m_choices[-1]:
            f.write('%.2f & \n' %psnr0_avg)
        else:
            f.write('%.2f & ' %psnr0_avg)
    for m in m_choices:
        psnr1 = []
        datapath = os.path.join('.',ssss,img_test,'m%d'%m)
        temp = [it for it in os.listdir(datapath) if it[-7:] == '.pickle']
        for t in temp:
            with open(os.path.join(datapath,t)) as ff:
                snr1, snr0 = pickle.load(ff)
            psnr1.append(snr1[thr[m]])
        psnr1_avg = np.mean(psnr1)
        if m == m_choices[-1]:
            f.write('%.2f \\\\ \n' %psnr1_avg)
        else:
            f.write('%.2f & ' %psnr1_avg)
f.close()














