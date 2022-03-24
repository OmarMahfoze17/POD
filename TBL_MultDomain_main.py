from __future__ import division
from __future__ import absolute_import
import sys
from mpi4py import MPI
sys.path.append("/ccc/scratch/cont005/ra5138/mahfozeo/TBL_/POD/python/modred-master")
sys.path.append("/ccc/cont005/home/unilond/mahfozeo/f90nml")
import numpy as np
import modred as mr
import os
import struct
import f90nml
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import h5py
import pickle
import time
from future.builtins import range
from myFunc import getZModes, calWeights, Screen, getBlockRange
comm = MPI.COMM_WORLD
fScreen = open(f'./Screen_{MPI.COMM_WORLD.rank}', "w")
fScreen.flush()

path='/ccc/scratch/cont005/ra5138/mahfozeo/TBL_/Re_1250/Lx_750/canon/FFT_z/'
MainOutDir='/ccc/scratch/cont005/ra5138/mahfozeo/TBL_/Re_1250/Lx_750/canon/FFT_z/POD_yp_40_step_4/'
#Y+=40
ix1_=[0  ,27  , 114, 201, 445, 690, 0  , 27 , 114, 201, 445  , 27 , 201, 0, 27, 114, 201]
ix2_=[1  ,28  , 115, 202, 446, 691, 27 , 114, 201, 445, 690  , 201, 690, 2, 30, 120, 250]
iy1_=[0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0    , 0  , 0  , 0, 0 , 0  , 0  ]
iy2_=[70 , 72 , 75 , 77 , 82 , 84 , 71 ,74  , 76 , 80 , 83   , 75 , 81 , 70, 72, 75, 78 ]

#Y+=180
#ix1_=[0  ,27  , 114, 201, 445, 690, 0  , 27 , 114, 201, 445, 27 , 201]
#ix2_=[1  ,28  , 115, 202, 446, 691, 27 , 114, 201, 445, 690, 201, 690]
#iy1_=[0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  ]
#iy2_=[252, 256, 264, 270, 279, 284, 254, 260, 267, 275, 282, 263, 277]

#Y=BL_thick
#ix1_=[0  ,27  , 114, 201, 445, 690, 0  , 27 , 114, 201, 445 , 27 , 201]
#ix2_=[1  ,28  , 115, 202, 446, 691, 27 , 114, 201, 445, 690 , 201, 690]
#iy1_=[0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0   , 0  , 0  ]
#iy2_=[261, 278, 316, 340, 383, 409, 270,297 , 328, 362, 396 , 309, 375]


zModes_t=range(0,64)
POD_mode_indices=range(5)
step=4
blckRng=getBlockRange(4000,160,step)
zModes=getZModes(zModes_t)
Screen(f'Rank {comm.rank} works on {len(zModes)} modes of {zModes}',All=True)
nBlock=range(0,160)
for ix1,ix2,iy1,iy2 in zip(ix1_,ix2_,iy1_,iy2_):
    subDomain=[ ix1, ix2, iy1, iy2]
    out_subDir = MainOutDir+'POD_sub_'+str(ix1)+'-'+str(ix2)+'-'+str(iy1)+'-'+str(iy2)+'/'
    os.system("mkdir "+out_subDir)
    Screen(f'Out put Dir: {out_subDir}',All=False)

nml = f90nml.read(path+"input.i3d")
nx = nml['BasicParam']['nx']
ny = nml['BasicParam']['ny']
nz = nml['BasicParam']['nz']
lx = nml['BasicParam']['xlx']
ly = nml['BasicParam']['yly']
lz = nml['BasicParam']['zlz']
IX_1 = nml['InOutParam']['ivs']
IX_2 = nml['InOutParam']['ive']
istret = nml['BasicParam']['istret']
if istret != 0:
    yfile = open(path+"yp.dat", "r")
    y = np.loadtxt(yfile)
else:
    y = np.linspace(0, ly, ny)

Domain=[IX_1, IX_2, 0, ny]

for k in zModes:
    start=time.time()
    Screen('------------------------------------------')
    Screen(f'Rank {str(comm.rank)} : Working on Spanwise mode  {k}/{zModes[-1]}.')
    Screen('------------------------------------------')

    t0 =time.time()
    ################## Read data files #######################
    for n in nBlock:
        fileName=path+'Data_'+ str(n).zfill(5)+'/'+str(k).zfill(5)
        f = h5py.File(fileName, 'r')
        #print(f'Rank {str(comm.rank)}: Zmode {k}, Reading {n}/{nBlock[-1]}', end="\r", flush=True)
        fScreen.write(f'Rank {str(comm.rank)}: Zmode {k}, Reading {n}/{nBlock[-1]} \n')
        fScreen.flush()
        if n==nBlock[0]:
            ux_temp =f['ux'][()][:,:,blckRng[n]]
            uy_temp =f['uy'][()][:,:,blckRng[n]]
            uz_temp =f['uz'][()][:,:,blckRng[n]]
        else:
            ux_temp =np.append(ux_temp ,f['ux'][()][:,:,blckRng[n]], axis=2)
            uy_temp =np.append(uy_temp ,f['uy'][()][:,:,blckRng[n]], axis=2)
            uz_temp =np.append(uz_temp ,f['uz'][()][:,:,blckRng[n]], axis=2)
    #    print(ux_temp.shape)
    Screen('')
    t1=time.time()
    Screen(f'Rank {str(comm.rank)}: Reading time blck {k}/{zModes[-1]} is {np.round((t1-t0)/60,2)} m')
    Screen(f'Rank {str(comm.rank)}: Computing POD of mode  {k}/{zModes[-1]}')
    for ix1,ix2,iy1,iy2 in zip(ix1_,ix2_,iy1_,iy2_):
    ################# work on the sub domains ############################
        out_subDir = MainOutDir+'POD_sub_'+str(ix1)+'-'+str(ix2)+'-'+str(iy1)+'-'+str(iy2)+'/'
        out_dir=out_subDir+'POD_zMode'+str(k).zfill(5)
        NY=iy2-iy1
        NX=ix2-ix1
        subDomain=[ ix1, ix2, iy1, iy2]
        _, weights =calWeights(y,subDomain)
        ux=np.reshape(ux_temp[ix1:ix2,iy1:iy2,:],(NX*NY,-1), order='F')
        uy=np.reshape(uy_temp[ix1:ix2,iy1:iy2,:],(NX*NY,-1), order='F')
        uz=np.reshape(uz_temp[ix1:ix2,iy1:iy2,:],(NX*NY,-1), order='F')
        data=np.vstack((ux,uy,uz))
        del ux,uy,uz
        DataShape=data.shape

        t2=time.time()
        #########################  Compute POD ###################################
        POD_res = mr.compute_POD_arrays_snaps_method(data,mode_indices=POD_mode_indices, inner_product_weights=weights)
        t3=time.time()
        Screen(f'Rank {str(comm.rank)}: Finish Computing POD mode {k}/{zModes[-1]}, SubDomain {subDomain}. Time {np.round((t3-t2)/60,2)} m')
        # Screen(POD_res.eigvals[0:5]/np.sum(POD_res.eigvals)*100)

#########################   Export the resulsts ##################
        fID=h5py.File(out_dir, 'w')
        fID.create_dataset('Domain', data=Domain)
        fID.create_dataset('subDomain', data=subDomain)
        fID.create_dataset('DataShape', data=DataShape)
        fID.create_dataset('StepSnap', data=step)
        fID.create_dataset('eigvals', data=POD_res.eigvals)
  #      fID.create_dataset('eigvecs', data=POD_res.eigvecs)
        fID.create_dataset('modes', data=POD_res.modes)
#        fID.create_dataset('proj_coeffs', data=POD_res.proj_coeffs)
 #       fID.create_dataset('correlation_array', data=POD_res.correlation_array)
        fID.close()
        del POD_res, data
    del ux_temp,  uy_temp, uz_temp
fScreen.close()
