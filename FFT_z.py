from __future__ import division
from __future__ import absolute_import
import sys
sys.path.append("/ccc/cont005/home/unilond/mahfozeo/f90nml")
import sys
from mpi4py import MPI
import numpy as np
import os
import struct
import f90nml
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import h5py
import time
from future.builtins import range
from myFunc import getZModes, getMean
#sys.stdout = open('./LOG', 'w')
from datetime import datetime

comm = MPI.COMM_WORLD


pathIn='/ccc/scratch/cont005/ra5138/mahfozeo/TBL_/Re_1250/Lx_750/canon/';
pathOut=pathIn+'FFT_z/';
if comm.rank==0:
    print('pathIn : ',pathIn)
    print('pathOut : ',pathOut)
    os.system("mkdir "+pathOut)
#    sys.stdout.flush()
comm.Barrier()
fSt=1;
fEnd=4001;
step=1;
Ns=25;
nMeanf=4
zModes=range(0,64)

nSnaps=int((fEnd-fSt)/step+1)
nBlock=range(int(nSnaps/Ns))
nBlock_MPI=getZModes(nBlock)
Shift=Ns*nBlock_MPI[0]
Nb=0
# fEnd=len(nBlock)*Ns*step+fSt
nFile=range(fSt,fEnd,step)
nFile_MPI=getZModes(nFile)
print(f'rank {comm.rank}, number of blocks {len(nBlock_MPI)} the blocks are {nBlock_MPI}, Shift {Shift}, number of files {len(nFile_MPI)}, the files are {nFile_MPI}')
#sys.stdout.flush()

# path='/media/omar/MyBook_3/Data/Channel/POD/DNS/Re_180_highRes/POD_1D/H5/'

nml = f90nml.read(pathIn+"input.i3d")
nx = nml['BasicParam']['nx']
ny = nml['BasicParam']['ny']
nz = nml['BasicParam']['nz']
lx = nml['BasicParam']['xlx']
ly = nml['BasicParam']['yly']
lz = nml['BasicParam']['zlz']
IX_1 = nml['InOutParam']['ivs']
IX_2 = nml['InOutParam']['ive']
nx_snap=IX_2-IX_1+1;
istret = nml['BasicParam']['istret']
if istret != 0:
    yfile = open(pathIn+"yp.dat", "r")
    y = np.loadtxt(yfile)
else:
    y = np.linspace(0, ly, ny)
x = np.linspace(0, lx, nx)
z = np.linspace(0, lz, nz)

if comm.rank==0:
    print('Reading Mean Velocity field')
#sys.stdout.flush()
if comm.rank==0:
    umean, vmean, wmean= getMean(pathIn,[nx,ny,nz],nMeanf)
else:
    umean=np.zeros((nx,ny,nz), dtype=np.float32)
    vmean=np.zeros((nx,ny,nz), dtype=np.float32)
    wmean=np.zeros((nx,ny,nz), dtype=np.float32)

comm.Bcast(umean, root=0)
comm.Bcast(vmean, root=0)
comm.Bcast(wmean, root=0)
if comm.rank==0:
    print('Start of FFT')
comm.Barrier()
#sys.stdout.flush()
uxn=np.zeros((nx_snap,ny,len(zModes),Ns),dtype=np.singlecomplex);
uyn=np.zeros((nx_snap,ny,len(zModes),Ns),dtype=np.singlecomplex);
uzn=np.zeros((nx_snap,ny,len(zModes),Ns),dtype=np.singlecomplex);

t0=time.time()
for n in nFile_MPI:

    N=int((n-fSt+step)/step)-1-Shift
    # N=int(n)-1-Shift
    print(f'rank {comm.rank} ,  File n {n}/{nFile_MPI[-1]}, subfile N {N}')
 #   sys.stdout.flush()
    ufile = open(pathIn+"ux"+str(n).zfill(5), "rb")
    vfile = open(pathIn+"uy"+str(n).zfill(5), "rb")
    wfile = open(pathIn+"uz"+str(n).zfill(5), "rb")
    UX=np.reshape(np.fromfile(ufile, dtype=np.float32),(nx_snap,ny,-1),order='F')-umean[IX_1:IX_2+1,:,:]
    UY=np.reshape(np.fromfile(vfile, dtype=np.float32),(nx_snap,ny,-1),order='F')-vmean[IX_1:IX_2+1,:,:]
    UZ=np.reshape(np.fromfile(wfile, dtype=np.float32),(nx_snap,ny,-1),order='F')-wmean[IX_1:IX_2+1,:,:]
    ufile.close();vfile.close();wfile.close();
    UX = np.singlecomplex(np.fft.fft(UX,axis=2));
    UY = np.singlecomplex(np.fft.fft(UY,axis=2));
    UZ = np.singlecomplex(np.fft.fft(UZ,axis=2));
    # UX=umean[IX_1:IX_2+1,:,:] #
    # UY=umean[IX_1:IX_2+1,:,:]#
    # UZ=umean[IX_1:IX_2+1,:,:]#

    uxn[:,:,:,N]=UX[:,:,zModes]
    uyn[:,:,:,N]=UY[:,:,zModes]
    uzn[:,:,:,N]=UZ[:,:,zModes]

    if np.mod(N+1,Ns)==0:
        t1=time.time()
        Nb+=1
        Shift+=Ns
        subPath=pathOut+'Data_'+ str(nBlock_MPI[Nb-1]).zfill(5)+'/'
        os.system("mkdir "+subPath)
        for k in zModes:
            fileName=(subPath+str(k).zfill(5))
            fID=h5py.File(fileName, 'w')
            fID.create_dataset('ux', data=uxn[:,:,k,:])
            fID.create_dataset('uy', data=uyn[:,:,k,:])
            fID.create_dataset('uz', data=uzn[:,:,k,:])
            fID.close()
        t2=time.time()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
	
        print(f'Rank {comm.rank} : Reading and FFT time = {np.round(t1-t0,2)}. \n   Writing time of blck{ nBlock_MPI[Nb-1]}/{nBlock_MPI[-1]} = {np.round(t2-t1,2)} \n Clock {current_time}')
  #      sys.stdout.flush()
        t0=time.time()
    # exit()

print(Nb)

