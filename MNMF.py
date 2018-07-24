# - coding: utf-8 -
import time
import datetime
import sys
import os
import re
import shutil
from collections import defaultdict
# import warnings
# warnings.simplefilter(action='ignore',category=FutureWarning)

import click
import numpy as np
import numpy.linalg as LA
import numpy.matlib
# sshでログインした時
import matplotlib as mpl
mpl.use('Agg')
# Matplotlibのバックエンドを変える
import matplotlib.pyplot as plt

from numpy.random import rand
from scipy.cluster.hierarchy import linkage,fcluster
from scipy import signal
from scipy.io import wavfile
import pandas as pd

from iohandler import read_wavs, write_wavs
from stft import stft,istft
import ILRMA
from evaluater import Evaluate
from evaluater import snr


eps = 1e-8


class NanInfException(Exception):
    """
    NaNやInfが出たときに発生させる例外
    """
    def __init__(self,instr,ptn):
        self._instr,self._ptn = instr,ptn
    def __str__(self):
        return '{}: {}'.format(self._ptn,self._instr)

def nanorinf(value,string):
    if np.any(np.isnan(value)):
        raise NanInfException(string,'NaN')
    if np.any(np.isinf(value)):
        raise NanInfException(string,'Inf')
    # if LA.cond(value) > 1e10:
    #     raise NanInfException('condition number is too large.','Inf')

def nanorinfcost(value,string):
    print(np.max(LA.cond(value)))
        
    if np.any(np.isnan(value)):
        raise NanInfException(string,'NaN')
    if np.any(np.isinf(value)):
        raise NanInfException(string,'Inf')
    # if LA.cond(value) > 1e10:
    #     raise NanInfException('condition number is too large.','Inf')

class MNMF(object):
    def __init__(self,
                signals,
                N,
                K,
                mode=3, # modeling way of noise spatial covariance
                n_iter=100,
                refMic=1, # reference mic for evaluation
                fftSize = 1024, # window length in STFT
                shiftSize = 512, # shift length in STFT TODO <- ILRMA.py ではset_evaluater以外では用いられていないか?
                draw = True, # plot cost function values or not(true or false)
                dump_step = 10, # 何回に1回Evaluaterを通すか
                dump_voice = False,
                dump_spectrogram = False,
                dump_sdr = True,
                refimg=None,
                output_path='output/',
                Xobs=None,
                ilrmares=None,
                **params
                ):

        self.signals = signals # I x J x M
        self.N = N
        self.K = K
        self.mode = mode # 分割関数の有無
        if mode not in [1,2]:
            raise ValueError
        self.n_iter = n_iter
        self.refMic = refMic
        self.fftSize = fftSize
        self.shiftSize = shiftSize
        self.draw = draw
        self.dump_step = dump_step
        self.dump_voice = dump_voice
        self.dump_spectrogram = dump_spectrogram
        self.dump_sdr = dump_sdr
        self.refimg = refimg
        self.output_path = output_path
        self.Xobs = Xobs
        self.ilrmares = ilrmares

        self.evaluater = None
        self.Xinput = None
        self.I,self.J,self.M = None,None,None
        self.H = None
        self.T = None
        self.V = None
        self.Z = None
        self.Xhat = None

        # optional argument
        self.R_N_real = None
        self.normalization = None
        self.TV = None
        for key,value in params['params'].items():
            if key == 'R_N_real':
                self.R_N_real = value
            if key == 'normalization':
                self.normalization = value
            if key == 'TV':
                self.TV = value

    def init_param(self):
        I,J,M = self.signals.shape # I: Frequency, J: Time, M: Channel
        N = self.N
        K = self.K
        self.I,self.J,self.M = I,J,M
        
        self.Xinput = np.zeros(I,J,M,M,dtype=np.complex128)
        for i in range(I):
            for j in range(J):
                self.Xinput[i,j] = np.outer(self.signals[i,j],self.signals[i,j].conjugate())
        
        if self.mode == 1 # 分割関数有り
            self.H = np.tile(np.sqrt(np.eye(M)/M)[np.newaxis,np.newaxis,:,:],[I,N,:,:]) # I x N x M x M
        else:
            raise NotImplementedError

        self.T = np.maximum(np.random.randn(I,K),eps)
        self.V = np.maximum(np.random.randn(K,J),eps)

        if self.mode == 1:
            varZ = 0.01
            self.Z = varZ * np.random.randn(K,N) + 1./N
            self.Z = self.Z / np.tile(np.sum(self.Z,axis=1)[:,np.newaxis],[:,N]) # sum_n z_kn=1

        self.Xhat = np.zeros(I,J,M,M,dtype=np.complex128)
        self.calc_Xhat()

        self.cost = np.zeros(self.n_iter+1)


    def iterativeUpdate(self):
        try:
            # modeによる切り替えは各関数の中で行う
            # Update T
            self.calc_Xhi()
            self.calc_XhiXXhi()
            self.update_T()
            self.calc_Xhat()

            # Update V
            self.calc_Xhi()
            self.calc_XhiXXhi()
            self.update_V()
            self.calc_Xhat()

            # Update Z
            self.calc_Xhi()
            self.calc_XhiXXhi()
            self.update_Z()
            self.calc_Xhat()

            # Update H
            self.calc_Xhi()
            self.calc_XhiXXhi()
# 分割関数なしならHのスケールをTのスケールで打ち消すことが出来る
            self.calc_H()
            self.calc_Xhat()

        except LA.linalg.LinAlgError as e:
            print(e.args[0])
            return False
        except NanInfException as e:
            print(e)
            return False
        else:
            return True


    def calc_Xhat(self):
        if self.mode == 1:
            for m1 in range(self.M):
                for m2 in range(self.M):
                    Hmm = self.H[:,:,m1,m2]
                    self.Xhat[:,:,m1,m2] = ((Hmm@np.conjugate(self.Z.T))*self.T)@self.V
        else:
            raise NotImplementedError

    def calc_Xhi(self):
        for i in range(self.I):
            for j in range(self.J):
                self.Xhi[i,j,:,:] = LA.inv(self.Xhat[i,j,:,:])

    def calc_XhiXXhi(self):
        for i in range(self.I):
            for j in range(self.J):
                self.XhiXXhi[i,j,:,:] = self.Xhi[i,j,:,:]@self.X[i,j,:,:]@self.Xhi[i,j,:,:]

    def update_T(self):
        if self.mode == 1:
            Tnume = np.zeros(self.I,self.K)
            Tdeno = np.zeros(self.I,self.K)
            for m1 in range(self.M):
                for m2 in range(self.M):
                    Tnume += np.real( (self.XhiXXhi[:,:,m1,m2]@self.V.T)*(np.conjugate(self.H[:,:,m1,m2])@np.conjugate(self.Z.T)) )
                    Tdeno += np.real( (self.Xhi[:,:,m1,m2]@self.V.T)*(np.conjugate(self.H[:,:,m1,m2])@np.conjugate(self.Z.T)) )
            self.T = self.T * np.maximum(np.sqrt(Tnume/T.deno),eps)

        else:
            raise NotImplementedError

    def update_V(self):
        if self.mode == 1:
            Vnume = np.zeros(self.K,self.J)
            Vdeno = np.zeros(self.K,self.J)
            for m1 in range(self.M):
                for m2 in range(self.M):
                    Vnume += np.real( np.conjugate((self.H[:,:,m1,m2]@np.conjugate(self.Z.T))*self.T).T@self.XhiXXhi[:,:,m1,m2] )
                    Vdeno += np.real( np.conjugate((self.H[:,:,m1,m2]@np.conjugate(self.Z.T))*self.T).T@self.Xhi[:,:,m1,m2] )
            self.V = self.V * np.maximum(np.sqrt(Vnume/Vdeno),eps)
        else:
            raise NotImplementedError

    def update_Z(self):
        if self.mode == 1:
            Znume = np.zeros(self.K,self.N)
            Zdeno = np.zeros(self.K,self.N)
            for m1 in range(self.M):
                for m2 in range(self.M):
                    Znume += np.real( np.conjugate(self.XhiXXhi[:,:,m1,m2]@self.V.T*self.T).T@self.H[:,:,m1,m2] )
                    Zdeno += np.real( np.conjugate(self.Xhi[:,:,m1,m2]@self.V.T*self.T).T@self.H[:,:,m1,m2] )
            self.Z = self.Z * np.sqrt(Znume/Zdeno)
            self.Z = self.Z / np.tile(np.sum(self.Z,axis=1)[:,np.newaxis],[:,self.N])
            self.Z = np.maximum(self.Z,eps)
        else:
            pass

    def calc_H(self):
        if self.mode == 1:
            X = np.reshape(self.XhiXXhi,[self.I,self.J,self.M*self.M])
            Y = np.reshape(self.Xhi,[self.I,self.J,self.M*self.M])
            for n in range(self.N):
                for i in range(self.I):
                    ZTV = (T[i,:]*Z[:,n].T)@V # shape: J
                    A = np.reshape(Y[i,:,:].T*ZTV,[self.M,self.M])
                    B = np.reshape(X[i,:,:].T*ZTV,[self.M,self.M])
                    Hin = np.reshape(self.H[i,n,:,:],[self.M,self.M])
                    C = Hin@B@Hin
                    AC = np.vstack((np.hstack((np.zeros(self.M),-1*A)),np.hstack((-1*C,np.zeros(self.M)))))
                    [eigVal, eigVec] = LA.eig(AC)
                    ind = np.where(eigVal<0)
                    F = np.squeeze(eigVec[:self.M,ind])
                    G = np.squeeze(eigVec[self.M:,ind])
                    Hin = G.dot(LA.inv(F))
                    Hin = (Hin+np.conjugate(Hin.T))/2
                    [eigVal, eigVec] = LA.eig(Hin)
                    eigVal = np.maximum(eigVal,eps)
                    Hin = eigVec*np.diag(eigVal).dot(LA.inv(eigVec))
                    Hin = (Hin+np.conjugate(Hin.T))/2
                    Hin += eps*np.eye(M)
                    H[i,n,:,:] = Hin/np.trace(Hin)
        else:
            raise NotImplementedError


    def cost_function(self,it):
        try:
            self.calc_Xhi()
            cost = 0
            XXhi = np.zeros(self.I,self.J,self.M,self.M,dtype=np.complex128)
            for i in range(self.I):
                for j in range(self.J):
                    XXhi[i,j,:,:] = self.X[i,j,:,:] @ self.Xhi[i,j,:,:]
                    cost += np.real(np.trace(XXhi[i,j,:,:])) - np.log(np.real(LA.det(XXhi[i,j,:,:]))) - self.M
            return cost
        except LA.linalg.LinAlgError as e:
            print(e.args[0])
            return np.inf
        except NanInfException as e:
            print(e)
            return np.inf

    def set_evaluater(self, signals):
        self.evaluater = Evaluate(signals, None, self.output_path+'wavs', self.shiftSize,self.refMic)

    def report_eval(self,it=0,last=False):
        simg_V = self.chatV*np.tile(self.a_V_ILRMA[:,np.newaxis,:],(1,self.J,1))[:,:,self.refMic]
        simg_N = self.chatN[:,:,self.refMic] # referenceは音声と雑音のsourceimgなので，chatNも1chのみ取り出す

        self.evaluater.eval_mir_duong(np.array([simg_V, simg_N]).transpose(1,2,0))

        if self.dump_voice:
            self.evaluater.dump_wav(simg_V, str(it)+"_voice")
            self.evaluater.dump_wav(simg_N, str(it)+"_noise")

        if self.dump_spectrogram:
            plt.figure()
            plt.pcolormesh(np.abs(simg_V)**0.3)
            plt.savefig('{}/figure/_out_specvoice_{}.eps'.format(self.output_path, it))
            plt.close()

        if self.dump_sdr and last:
            self.evaluater.dump_sdrplot(self.output_path,self.ilrma_res)
            self.best_sdrs = self.evaluater.best_sdrs()
            best_sdrs = self.best_sdrs
            with open("{}/bestSDRs.txt".format(self.output_path) ,"w") as f:
                f.write("ILRMA SDR,SIR,SAR\n")
                f.write("SDR: {}dB.\n".format(best_sdrs['SDRilrma']))
                f.write("SIR: {}dB.\n".format(best_sdrs['SIRilrma']))
                f.write("SAR: {}dB.\n".format(best_sdrs['SARilrma']))
                f.write("\n")
                f.write("best SDR,SIR,SAR\n")
                f.write("SDR: {}dB at {}itrs.\n".format(best_sdrs['SDR'],best_sdrs['SDRind']))
                f.write("SIR: {}dB at {}itrs.\n".format(best_sdrs['SIR'],best_sdrs['SIRind']))
                f.write("SAR: {}dB at {}itrs.\n".format(best_sdrs['SAR'],best_sdrs['SARind']))


    def iterate(self):
        self.init_param()
        # self.report_eval()
        for it in range(1,self.n_iter+1):
            ret = self.iterativeUpdate()
            if not ret:
                # iterativeUpdateでエラー発生時
                # self.evaluater.disable_res()
                return False

            if self.draw:
                self.cost[it-1] = self.cost_function(it)
                if self.cost[it-1] == np.inf:
                    self.evaluater.disable_res()
                    return False
                print("Iteration: {} : {}".format(it-1,self.cost[it-1]))

            if it % self.dump_step == 0:
                self.wiener_filter()
                self.report_eval(it,last=(it==self.n_iter))

        if self.draw:
            plt.figure()
            plt.plot(np.arange(self.n_iter),self.cost[:-1])
            plt.title('cost function')
            plt.savefig('{}figure/cost/mode{}.eps'.format(self.output_path,self.mode))
            plt.close()

        return True


    def wiener_filter(self):
        pass

    def main(self,**params):

        if self.Xobs is not None:
            X = self.Xobs
        else:
            mix = self.signals.sum(axis=0)
            X, window = stft(mix,self.fftSize,self.shiftSize)
        self.Xinput = X
        self.set_evaluater(self.refimg)
        if self.iterate():
            self.wiener_filter()
            simg_V = self.chatV * np.tile(self.a_V_ILRMA[:,np.newaxis,:],(1,self.J,1))[:,:,self.refMic]
            sepV = istft(simg_V,self.shiftSize)
            sepN = istft(self.chatN,self.shiftSize)
            return sepV, sepN
        else:
            return [None for _ in range(self.N)]


class ILRMADuong(object):
    def __init__(self,
                 snr_set,
                 noise_env,
                 mode,
                 index_v=3,
                 nb=10,
                 n_iter=50,
                 fftSize=4096,
                 shiftSize=2048,
                 date=None,
                 npyload=False,
                 **params
                 ):
        self.fftSize=fftSize
        self.shiftSize=shiftSize
        self.snr_set = snr_set
        self.noise_env = noise_env
        self.mode = mode
        self.index_v = index_v
        self.nb = nb
        self.n_iter = n_iter
        self.params = defaultdict(list)
        self.npyload = npyload
        self.normalization = 'None'

        if params is not None:
            for key,value in params['params'].items():
                if key == 'R_N_real' and value == 'True':
                    self.params['R_N_real'] = None
                if key == 'normalization':
                    self.params['normalization'] = value
                    self.normalization = value
                if key == 'TV' and value == 'True':
                    self.params['TV'] = value
                if key == 'TVBP' and value == 'True':
                    self.TVBP = True
        self.output_path = '/media/hdd/DuongPy/output_{0:02d}{1:02d}_{2:02d}{3:02d}/{4}/{5}/{6}/'.format(mo,da,ho,mi,noise_env,snr_set,self.normalization)


    def main(self):
        file_simg = ['input/voice/originalSource{}.wav'.format(i+1) for i in range(4)]
        file_noise = ['input/noise/{}/{}.wav'.format(self.noise_env,i+1) for i in range(4)]

        signals = [None,None]
        signals[0] = read_wavs(file_simg,16000)
        signals[1] = np.array(read_wavs(file_noise,16000))
        length = int(min(len(signals[0][0]),len(signals[1][0]))*0.9)
        sig = [[None,None,None,None],[None,None,None,None]]
        for i in range(len(signals[0])):
            sig[0][i] = signals[0][i][:length]
            sig[1][i] = signals[1][i][:length]
        sig = np.array(sig)
        sig[1] *= 10**((snr(sig[0],sig[1])-self.snr_set)/20)
        signals = np.array(sig,dtype=np.float64).transpose((0,2,1))
        print(signals.shape)
        sys.exit()


        if self.npyload == True:
            if not os.path.exists('npy/{}_{}_ilrmaW.npy'.format(self.noise_env,self.snr_set)):
                self.npyload = False
            elif not os.path.exists('npy/{}_{}_ilrmaR.npy'.format(self.noise_env,self.snr_set)):
                self.npyload = False
            elif not os.path.exists('npy/{}_{}_ilrmaXobs.npy'.format(self.noise_env,self.snr_set)):
                self.npyload = False
            else:
                w = np.load('npy/{}_{}_ilrmaW.npy'.format(self.noise_env,self.snr_set))
                Xobs = np.load('npy/{}_{}_ilrmaXobs.npy'.format(self.noise_env,self.snr_set))
                r = np.load('npy/{}_{}_ilrmaR.npy'.format(self.noise_env,self.snr_set))
        if self.npyload == False:
            ilrma = ILRMA.ILRMA(refMic=3,fs_resample=16000,files='vocal/bass',fftSize=self.fftSize,shiftSize=self.shiftSize,n_iter=50,nb=self.nb,TVBP=self.TVBP,dump_step=1,output_path='{}ILRMA'.format(self.output_path))
            ilrma.main(signals)
            np.save('npy/{}_{}_ilrmaW.npy'.format(self.noise_env,self.snr_set),ilrma.W)
            np.save('npy/{}_{}_ilrmaR.npy'.format(self.noise_env,self.snr_set),ilrma.R)
            np.save('npy/{}_{}_ilrmaXobs.npy'.format(self.noise_env,self.snr_set),ilrma.Xobs)
            w = np.load('npy/{}_{}_ilrmaW.npy'.format(self.noise_env,self.snr_set))
            Xobs = np.load('npy/{}_{}_ilrmaXobs.npy'.format(self.noise_env,self.snr_set))
            r = np.load('npy/{}_{}_ilrmaR.npy'.format(self.noise_env,self.snr_set))

        self.duong = DuongMod(signals=signals,W_ILRMA=w,R_ILRMA=r,mode=self.mode,n_iter=self.n_iter,index_v=self.index_v,draw=True,dump_step=1,fftSize=self.fftSize,shiftSize=self.shiftSize,refimg=signals,dump_voice=True,dump_spectrogram=False,dump_sdr=True,refMic=3,Xobs=Xobs,output_path = self.output_path,params=self.params)
        self.duong.main()

