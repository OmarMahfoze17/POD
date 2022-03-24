from mpi4py import MPI
import numpy as np
import f90nml
import h5py
import time
from future.builtins import range

def getZModes(zModes_t):
    comm = MPI.COMM_WORLD
    num_workers = comm.Get_size()
    iRank=comm.rank
    nZModes=int(np.round(len(zModes_t)/num_workers+.1))
    if iRank+1==num_workers:
        zModes=zModes_t[iRank*nZModes:]
    else:
        zModes=zModes_t[iRank*nZModes:nZModes*(iRank+1)]
    return zModes

def getMean(pathIn,Domain,nFile):
    nx=Domain[0];ny=Domain[1];nz=Domain[2];
    ufile = open(pathIn+"meanVal_"+str(1)+"/umean.dat", "rb")
    vfile = open(pathIn+"meanVal_"+str(1)+"/vmean.dat", "rb")
    wfile = open(pathIn+"meanVal_"+str(1)+"/wmean.dat", "rb")
    umean = np.fromfile(ufile, dtype=np.float32)
    vmean = np.fromfile(vfile, dtype=np.float32)
    wmean = np.fromfile(wfile, dtype=np.float32)
    for i in range(2,nFile+1):
        ufile = open(pathIn+"meanVal_"+str(i)+"/umean.dat", "rb")
        vfile = open(pathIn+"meanVal_"+str(i)+"/vmean.dat", "rb")
        wfile = open(pathIn+"meanVal_"+str(i)+"/wmean.dat", "rb")
        umean += np.fromfile(ufile, dtype=np.float32)
        vmean += np.fromfile(vfile, dtype=np.float32)
        wmean += np.fromfile(wfile, dtype=np.float32)
    ufile.close()
    vfile.close()
    wfile.close()

    umean= np.reshape(umean,(nx,ny,-1),order='F')
    vmean= np.reshape(vmean,(nx,ny,-1),order='F')
    wmean= np.reshape(wmean,(nx,ny,-1),order='F')

    umean=np.mean(umean,axis=2)
    vmean=np.mean(vmean,axis=2)
    wmean=np.mean(wmean,axis=2)
    nt=int(umean[0,-1])
    print('Nt = ',nt)
    umean/=nt;vmean/=nt;wmean/=nt
    # umean=np.repeat(umean,[5],axis=2)
    # umean=np.tile(umean,(nx,1,2))
    umean = np.repeat(umean[:, :, np.newaxis], nz, axis=2)
    vmean = np.repeat(vmean[:, :, np.newaxis], nz, axis=2)
    wmean = np.repeat(wmean[:, :, np.newaxis], nz, axis=2)
    return umean, vmean, wmean

