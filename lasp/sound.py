import copy
from copy import deepcopy
import fnmatch
from math import ceil
from numpy.fft import fftshift, ifftshift
import os
import subprocess
import wave
import struct
import hashlib
import h5py

import numpy as np
from scipy.io.wavfile import read as read_wavfile
from scipy.fftpack import fft, ifft, fftfreq, fft2, ifft2, dct
from scipy.signal import resample, firwin, filtfilt
from scipy.optimize import leastsq

import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import matplotlib.colors as pltcolors
import matplotlib.mlab as mlab

import colorsys
from lasp.signal import lowpass_filter, gaussian_window, correlation_function
from lasp.timefreq import gaussian_stft
from lasp.detect_peaks import *


class WavFile():
    """ Class for representing a sound and writing it to a .wav file """

    def __init__(self, file_name=None, log_spectrogram=True):

        self.log_spectrogram = log_spectrogram
        if file_name is None:
            self.sample_depth = 2  # in bytes
            self.sample_rate = 44100.0  # in Hz
            self.data = None
            self.num_channels = 1
        else:
            wr = wave.open(file_name, 'r')
            self.num_channels = wr.getnchannels()
            self.sample_depth = wr.getsampwidth()
            wr.close()

            self.sample_rate, data = read_wavfile(file_name)
            # If stereo make mono
            if self.num_channels == 1:
                self.data = data
            else:
                self.data = data.mean(axis=1)

        self.analyzed = False

    def to_wav(self, output_file, normalize=False, max_amplitude=32767.0):
        wf = wave.open(output_file, 'w')

        wf.setparams( (self.num_channels, self.sample_depth, self.sample_rate, len(self.data), 'NONE', 'not compressed') )
        # normalize the sample
        if normalize:
            nsound = ((self.data / np.abs(self.data).max())*max_amplitude).astype('int')
        else:
            nsound = self.data
        #print 'nsound.min=%d, max=%d' % (nsound.min(), nsound.max())
        hex_sound = [struct.pack('h', x) for x in nsound]
        wf.writeframes(''.join(hex_sound))
        wf.close()

    def analyze(self, min_freq=0, max_freq=None, spec_sample_rate=1000.0, freq_spacing=125.0, envelope_cutoff_freq=200.0, noise_level_db=80, rectify=True, cmplx=False):
        if self.analyzed:
            return

        self.data_t = np.arange(0.0, len(self.data), 1.0) / self.sample_rate

        #compute the temporal envelope
        self.envelope = temporal_envelope(self.data, self.sample_rate, envelope_cutoff_freq)

        #compute log power spectrum
        fftx = fft(self.data)
        ps_f = fftfreq(len(self.data), d=(1.0 / self.sample_rate))
        if max_freq == None:
            findx = (ps_f > min_freq) & (ps_f < np.inf)
        else:
            findx = (ps_f > min_freq) & (ps_f < max_freq)
        self.power_spectrum = np.log10(np.abs(fftx[findx]))
        self.power_spectrum_f = ps_f[findx]

        #estimate fundamental frequency from log power spectrum in the simplest way possible
        ps = np.abs(fftx[findx])
        peak_index = ps.argmax()
        try:
            self.fundamental_freq = self.power_spectrum_f[peak_index]
        except IndexError:
            print 'Could not identify fundamental frequency!'
            self.fundamental_freq = 0.0

        #compute log spectrogram
        t,f,spec,spec_rms = spectrogram(self.data, self.sample_rate, spec_sample_rate=spec_sample_rate,
                                        freq_spacing=freq_spacing, min_freq=min_freq, max_freq=max_freq)
        self.spectrogram_t = t
        self.spectrogram_f = f
        self.spectrogram = spec
        self.spectrogram_rms = spec_rms
        self.analyzed = True

    def reanalyze(self, min_freq=0, max_freq=None, spec_sample_rate=1000.0, freq_spacing=25.0, envelope_cutoff_freq=200.0, noise_level_db=80, rectify=True, cmplx=False):
        self.analyzed = False
        return self.analyze(min_freq, max_freq, spec_sample_rate, freq_spacing, envelope_cutoff_freq, noise_level_db, rectify, cmplx)

    def plot(self, fig=None, show_envelope=True, min_freq=0.0, max_freq=10000.0, colormap=cmap.gist_yarg, noise_level_db=80,
             start_time=0, end_time=np.inf):

        self.analyze(min_freq=min_freq, max_freq=max_freq, noise_level_db=noise_level_db)

        if show_envelope:
            spw_size = 15
            spec_size = 35
        else:
            spw_size = 25
            spec_size = 75

        raw_ti = (self.data_t > start_time) & (self.data_t < end_time)

        if fig is None:
            fig = plt.figure()
        gs = plt.GridSpec(100, 1)
        ax = fig.add_subplot(gs[:spw_size])
        plt.plot(self.data_t[raw_ti], self.data[raw_ti], 'k-')
        plt.axis('tight')
        plt.ylabel('Sound Pressure')

        s = (spw_size+5)
        e = s + spec_size
        ax = fig.add_subplot(gs[s:e])
        spec_ti = (self.spectrogram_t > start_time) & (self.spectrogram_t < end_time)
        plot_spectrogram(self.spectrogram_t[spec_ti], self.spectrogram_f, self.spectrogram[:, spec_ti], ax=ax, ticks=True, colormap=colormap, colorbar=False)

        if show_envelope:
            ax = fig.add_subplot(gs[(e+5):95])
            plt.plot(self.spectrogram_t, self.spectrogram_rms, 'g-')
            plt.xlabel('Time (s)')
            plt.ylabel('Envelope')
            plt.axis('tight')
            
class BioSound(object):
    """ Class for representing a communication sound using multiple feature spaces"""

    def __init__(self, soundWave=np.array(0.0), fs=np.array(0.0), emitter='Unknown', calltype = 'U' ):

        self.sound = soundWave  # sound pressure waveform 
        self.hashid = hashlib.md5(np.array_str(soundWave)).hexdigest()
        self.samprate = fs      # sampling rate
        self.emitter = emitter  # string for id of emitter
        self.type = calltype    # string for call type
        self.spectro = None    # Log spectrogram
        self.to = None         # Time scale for spectrogram
        self.fo = None         # Frequency scale for spectrogram
        self.mps = None        # Modulation Power Spectrum
        self.wf = None         # Spectral modulations
        self.wt = None         # Temporal modulations
        self.f0 = None         # time varying fundamental
        self.f0_2 = None       # time varying fundamental of second voice
        self.F1 = None         # time varying formant 1
        self.F2 = None         # time varying formant 2
        self.F3 = None         # time varying formant 3
        self.fund = None       # Average fundamental
        self.sal = None        # Average saliency
        self.fund2 = None      # Average fundamental of 2nd peak
        self.voice2percent = None # Average percent of presence of second peak
        self.maxfund = None
        self.minfund = None
        self.cvfund = None
        self.meanspect = None
        self.stdspect = None
        self.skewspect = None
        self.kurtosisspect = None
        self.entropyspect = None
        self.q1 = None
        self.q2 = None
        self.q3 = None
        self.meantime = None
        self.stdtime = None
        self.skewtime = None
        self.kurtosistime = None
        self.entropytime = None
        self.fpsd = None
        self.psd = None
        self.tAmp = None
        self.amp = None
        self.rms = None
        self.maxAmp = None
        
    def saveh5(self, fileName=None):
   # Save as an h5 file. Uses the hashid if fileName is not given
   # Not using attributes
   
        if fileName is None:
            fileName = '%s.h5' % self.hashid
            
        fid = h5py.File(fileName,'w')
        selfDict = vars(self)
        for varnames in selfDict:
            if selfDict[varnames] is not None:
                fid.create_dataset(varnames, data=np.array(selfDict[varnames]))
            
        fid.close()
        
    def readh5(self, fileName):
        
        fid = h5py.File(fileName, 'r')
        for varnames in fid.keys():
            setattr(self, varnames, np.array(fid[varnames]).squeeze())
        
        fid.close()
        
    def spectrum(self, f_high=10000):
    # Calculates power spectrum and features from power spectrum

    # Need to add argument for window size
    # f_high is the upper bound of the frequency for saving power spectrum
    # nwindow = (1000.0*np.size(soundIn)/samprate)/window_len
    # 
        Pxx, Freqs = mlab.psd(self.sound, Fs=self.samprate, NFFT=1024, noverlap=512)
    
        # Find quartile power
        cum_power = np.cumsum(Pxx)
        tot_power = np.sum(Pxx)
        quartile_freq = np.zeros(3, dtype = 'int')
        quartile_values = [0.25, 0.5, 0.75]
        nfreqs = np.size(cum_power)
        iq = 0
        for ifreq in range(nfreqs):
            if (cum_power[ifreq] > quartile_values[iq]*tot_power):
                quartile_freq[iq] = ifreq
                iq = iq+1
                if (iq > 2):
                    break
                 
        # Find skewness, kurtosis and entropy for power spectrum below f_high
        ind_fmax = mlab.find(Freqs > f_high)[0]
    
        # Description of spectral shape
        spectdata = Pxx[0:ind_fmax]
        freqdata = Freqs[0:ind_fmax]
        spectdata = spectdata/np.sum(spectdata)
        meanspect = np.sum(freqdata*spectdata)
        stdspect = np.sqrt(np.sum(spectdata*((freqdata-meanspect)**2)))
        skewspect = np.sum(spectdata*(freqdata-meanspect)**3)
        skewspect = skewspect/(stdspect**3)
        kurtosisspect = np.sum(spectdata*(freqdata-meanspect)**4)
        kurtosisspect = kurtosisspect/(stdspect**4)
        entropyspect = -np.sum(spectdata*np.log2(spectdata))/np.log2(ind_fmax)
 
        # Storing the values       
        self.meanspect = meanspect
        self.stdspect = stdspect
        self.skewspect = skewspect
        self.kurtosisspect = kurtosisspect
        self.entropyspect = entropyspect
        self.q1 = Freqs[quartile_freq[0]]
        self.q2 = Freqs[quartile_freq[1]]
        self.q3 = Freqs[quartile_freq[2]]
        self.fpsd = freqdata
        self.psd = spectdata
        
    def spectroCalc(self, spec_sample_rate=1000, freq_spacing = 50, min_freq=0, max_freq=10000, noise_level_db=80):
        # Calculates the spectrogram in dB
        t,f,spec,spec_rms = spectrogram(self.sound, self.samprate, spec_sample_rate=spec_sample_rate,
                                        freq_spacing=freq_spacing, min_freq=min_freq, max_freq=max_freq,
                                        cmplx=True)
        self.to = t
        self.fo = f
        self.spectro = 20*np.log10(np.abs(spec))

    def mpsCalc(self, window=None, Norm=True): 
        
        if self.spectro == None:
            self.spectroCalc()
            
        wf, wt, mps_powAvg = mps(self.spectro, self.fo, self.to, window=window, Norm=Norm)
        self.mps = mps_powAvg        # Modulation Power Spectrum
        self.wf = wf         # Spectral modulations
        self.wt = wt
        
        
    def ampenv(self):
    # Calculates the amplitude enveloppe and related parameters
    
        (amp, tdata)  = temporal_envelope(self.sound, self.samprate, cutoff_freq=20, resample_rate=1000)
        
        # Here are the parameters
        ampdata = amp/np.sum(amp)
        meantime = np.sum(tdata*ampdata)
        stdtime = np.sqrt(np.sum(ampdata*((tdata-meantime)**2)))
        skewtime = np.sum(ampdata*(tdata-meantime)**3)
        skewtime = skewtime/(stdtime**3)
        kurtosistime = np.sum(ampdata*(tdata-meantime)**4)
        kurtosistime = kurtosistime/(stdtime**4)
        indpos = mlab.find(ampdata>0)
        entropytime = -np.sum(ampdata[indpos]*np.log2(ampdata[indpos]))/np.log2(np.size(indpos))
        
        self.meantime = meantime   
        self.stdtime = stdtime
        self.skewtime = skewtime
        self.kurtosistime = kurtosistime
        self.entropytime = entropytime
        self.tAmp = tdata
        self.amp = amp
        self.maxAmp = max(amp)
        
    def fundest(self, maxFund = 1500, minFund = 300, lowFc = 200, highFc = 6000, minSaliency = 0.5):
    # Calculate the fundamental, the formants and parameters related to these
    
        sal, fund, fund2, form1, form2, form3, lenfund = fundEstimator(self.sound, self.samprate, self.to, debugFig = 0, maxFund = maxFund, minFund = minFund, lowFc = lowFc, highFc = highFc, minSaliency = minSaliency)
        goodFund = fund[~np.isnan(fund)]
        goodSal = sal[~np.isnan(sal)]
        goodFund2 = fund2[~np.isnan(fund2)]
        if np.size(goodFund) > 0 :
            meanfund = np.mean(goodFund)
        else:
            meanfund = None
        meansal = np.mean(goodSal)
        if np.size(goodFund2)> 0:
            meanfund2 = np.mean(goodFund2)
        else:
            meanfund2 = None
    
        if np.size(goodFund) == 0 or np.size(goodFund2) == 0:
            fund2prop = 0.0
        else:
            fund2prop = np.float(np.size(goodFund2))/np.float(np.size(goodFund))
            
        self.f0 = fund         # time varying fundamental
        self.f0_2 = fund2        # time varying fundamental of second voice
        self.F1 = form1         # time varying formant 1
        self.F2 = form2         # time varying formant 2
        self.F3 = form3         # time varying formant 3
        self.fund = meanfund       # Average fundamental
        self.sal = meansal        # Average saliency
        self.fund2 = meanfund2      # Average fundamental of 2nd peak
        self.voice2percent = fund2prop*100 # Average percent of presence of second peak
        if np.size(goodFund) > 0 :
            self.maxfund = np.max(goodFund)
            self.minfund = np.min(goodFund)
            self.cvfund = np.std(goodFund)/meanfund
            
    def play(self):
    # Plays the sound
        play_sound_array(self.sound*(2**15), self.samprate)
            
    def plot(self, DBNOISE=50, f_low=250, f_high=10000):
    # Plots a biosound in figures 1, 2, 3
    
        # Ploting Variables
        soundlen = np.size(self.sound)
        t = np.array(range(soundlen))
        t = t*(1000.0/self.samprate)

        # Plot the oscillogram + spectrogram
        plt.figure(1)
        plt.clf()
        # mngr = plt.get_current_fig_manager()
        # mngr.window.setGeometry(0, 260, 640, 545)

        
        # The oscillogram
        plt.axes([0.1, 0.75, 0.85, 0.20])      
        plt.plot(t,self.sound, 'k')
        # plt.xlabel('Time (ms)')
        plt.xlim(0, t[-1])               
        # Plot the amplitude enveloppe  
        if self.tAmp != None :      
            plt.plot(self.tAmp*1000.0, self.amp, 'r', linewidth=2)
      
        # Plot the spectrogram
        plt.axes([0.1, 0.1, 0.85, 0.6])
        spec_colormap()   # defined in sound.py
        cmap = plt.get_cmap('SpectroColorMap')
        
        if self.spectro != None :
            soundSpect = self.spectro
            if soundSpect.shape[0] == self.to.size:
                soundSpect = np.transpose(soundSpect)
            maxB = soundSpect.max()
            minB = maxB-DBNOISE
            soundSpect[soundSpect < minB] = minB
            plt.imshow(soundSpect, extent = (self.to[0]*1000, self.to[-1]*1000, self.fo[0], self.fo[-1]), aspect='auto', interpolation='nearest', origin='lower', cmap=cmap, vmin=minB, vmax=maxB)
        
        plt.ylim(f_low, f_high)
        plt.xlim(0, t[-1])
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (ms)')
                     
    # Plot the fundamental on the same figure
        if self.f0 != None :
            plt.plot(self.to*1000.0, self.f0, 'k', linewidth=3)
            plt.plot(self.to*1000.0, self.f0_2, 'm', linewidth=3)
            plt.plot(self.to*1000.0, self.F1, 'r--', linewidth=3)
            plt.plot(self.to*1000.0, self.F2, 'w--', linewidth=3)
            plt.plot(self.to*1000.0, self.F3, 'b--', linewidth=3)
        plt.show()
           
    # Plot Power Spectrum
        plt.figure(2)
        plt.clf()
        # mngr = plt.get_current_fig_manager()
        # mngr.window.setGeometry(650, 260, 640, 545)
        if self.psd != None :
            plt.plot(self.fpsd, self.psd, 'k-') 
            plt.xlabel('Frequency Hz')
            plt.ylabel('Power Linear')
        
            xl, xh, yl, yh = plt.axis()
            xl = 0.0
            xh = f_high
            plt.axis((xl, xh, yl, yh))
            plt.plot([self.q1, self.q1], [yl, yh], 'k--')
            plt.plot([self.q2, self.q2], [yl, yh], 'k--')
            plt.plot([self.q3, self.q3], [yl, yh], 'k--')
        
            F1Mean = self.F1[~np.isnan(self.F1)].mean()
            F2Mean = self.F2[~np.isnan(self.F2)].mean()
            F3Mean = self.F3[~np.isnan(self.F3)].mean()
            plt.plot([F1Mean, F1Mean], [yl, yh], 'r--', linewidth=2.0)
            plt.plot([F2Mean, F2Mean], [yl, yh], 'c--', linewidth=2.0)
            plt.plot([F3Mean, F3Mean], [yl, yh], 'b--', linewidth=2.0)
            plt.show()
  
    # Table of results
        plt.figure(3)
        plt.clf()
        # mngr = plt.get_current_fig_manager()
        # mngr.window.setGeometry(320, 10, 640, 250)
        textstr = '%s  %s' % (self.emitter, self.type)
        plt.text(0.4, 1.0, textstr)
        if self.fund is not None:
            if self.fund2 is not None:
                textstr = 'Mean Fund = %.2f Hz Mean Saliency = %.2f Mean Fund2 = %.2f PF2 = %.2f%%' % (self.fund, self.sal, self.fund2, self.voice2percent)
            else:
                textstr = 'Mean Fund = %.2f Hz Mean Saliency = %.2f No 2nd Voice Detected' % (self.fund, self.sal)
        else:
            textstr = 'Fundamental is ill defined'
        plt.text(-0.1, 0.8, textstr)
        if self.fund is not None:
            textstr = '   Max Fund = %.2f Hz, Min Fund = %.2f Hz, CV = %.2f' % (self.maxfund, self.minfund, self.cvfund) 
            plt.text(-0.1, 0.7, textstr)
        textstr = 'Mean Spect = %.2f Hz, Std Spect= %.2f Hz' % (self.meanspect, self.stdspect)
        plt.text(-0.1, 0.6, textstr)
        textstr = '   Skew = %.2f, Kurtosis = %.2f Entropy=%.2f' % (self.skewspect, self.kurtosisspect, self.entropyspect)
        plt.text(-0.1, 0.5, textstr)
        textstr = '   Q1 F = %.2f Hz, Q2 F= %.2f Hz, Q3 F= %.2f Hz' % (self.q1, self.q2, self.q3 )
        plt.text(-0.1, 0.4, textstr)
        textstr = '   For1 = %.2f Hz, For2 = %.2f Hz, For3= %.2f Hz' % (F1Mean, F2Mean, F3Mean )
        plt.text(-0.1, 0.3, textstr)
        textstr = 'Mean Time = %.2f s, Std Time= %.2f s' % (self.meantime, self.stdtime)
        plt.text(-0.1, 0.2, textstr)
        textstr = '   Skew = %.2f, Kurtosis = %.2f Entropy=%.2f' % (self.skewtime, self.kurtosistime, self.entropytime)
        plt.text(-0.1, 0.1, textstr)
        textstr = 'RMS = %.2f, Max Amp = %.2f' % (self.rms, self.maxAmp)
        plt.text(-0.1, 0.0, textstr)
        
        plt.axis('off')        
        plt.show()
        plt.pause(1)   # To flush the plots?



def spec_colormap():
# Makes the colormap that we like for spectrograms

    cmap = np.zeros((64,3))
    cmap[0,2] = 1.0

    for ib in range(21):
        cmap[ib+1,0] = (31.0+ib*(12.0/20.0))/60.0
        cmap[ib+1,1] = (ib+1.0)/21.0
        cmap[ib+1,2] = 1.0

    for ig in range(21):
        cmap[ig+ib+1,0] = (21.0-(ig)*(12.0/20.0))/60.0
        cmap[ig+ib+1,1] = 1.0
        cmap[ig+ib+1,2] = 0.5+(ig)*(0.3/20.0)

    for ir in range(21):
        cmap[ir+ig+ib+1,0] = (8.0-(ir)*(7.0/20.0))/60.0
        cmap[ir+ig+ib+1,1] = 0.5 + (ir)*(0.5/20.0)
        cmap[ir+ig+ib+1,2] = 1

    for ic in range(64):
        (cmap[ic,0], cmap[ic,1], cmap[ic,2]) = colorsys.hsv_to_rgb(cmap[ic,0], cmap[ic,1], cmap[ic,2])
    
    spec_cmap = pltcolors.ListedColormap(cmap, name=u'SpectroColorMap', N=64)
    plt.register_cmap(cmap=spec_cmap)

def plot_spectrogram(t, freq, spec, ax=None, ticks=True, fmin=None, fmax=None, colormap=None, colorbar=True, log = True, dBNoise = 50):
    
    if colormap == None:
        spec_colormap()
        colormap = plt.get_cmap('SpectroColorMap')
        
    if ax is None:
        ax = plt.gca()

    if fmin is None:
        fmin = freq.min()
    if fmax is None:
        fmax = freq.max()

    ex = (t.min(), t.max(), freq.min(), freq.max())
    plotSpect = np.abs(spec)
    
    
    if log == True and dBNoise is not None:
        plotSpect = 20*np.log10(plotSpect)
        maxB = plotSpect.max()
        minB = maxB-dBNoise
    else:
        if dBNoise is not None:
            maxB = 20*np.log10(plotSpect.max())
            minB = ((maxB-dBNoise)/20)**10
        else:
            maxB = plotSpect.max()
            minB = plotSpect.min()

    plotSpect[plotSpect < minB] = minB
                
    iax = ax.imshow(plotSpect, aspect='auto', interpolation='nearest', origin='lower', extent=ex, cmap=colormap, vmin=minB, vmax=maxB)
    plt.ylim(fmin, fmax)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')

    if colorbar:
        plt.colorbar(iax)


def play_sound(file_name):
    """ Install sox to get this to work: http://sox.sourceforge.net/ """
    subprocess.call(['play', file_name])

def play_wavfile(filename):

    import pyaudio

    chunk_size = 1024

    wf = wave.open(filename, "r")
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True)

    data = wf.readframes(chunk_size)

    while data != '':
        stream.write(data)
        data = wf.readframes(chunk_size)

    wf.close()
    stream.stop_stream()
    stream.close()
    p.terminate()

def play_sound_array(data, sample_rate):
    ''' Requires pyaudio package. Can be downloaded here
    http://people.csail.mit.edu/hubert/pyaudio/
    '''

    import pyaudio

    # Only play one channel
    if len(data.shape) > 1:
        data = np.mean(data, axis=np.argmin(data.shape))

    data = data.astype('int16')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(2),
        channels=1,
        rate=int(sample_rate),
        output=True)

    stream.write(data.tostring())
    stream.stop_stream()
    stream.close()
    p.terminate()


def spectrogram(s, sample_rate, spec_sample_rate, freq_spacing, min_freq=0, max_freq=None, nstd=6,  cmplx = True):
    """
        Given a sound pressure waveform, s, compute the complex spectrogram. 
        See documentation on gaussian_stft for additional details.

        Returns: 
          t, freq, timefreq, rms
              t: array of time values to use as x axis
              freq: array of frequencies to use as y axis
              timefreq: the spectrogram (a time-frequency represnetaion)
              rms : the time varying average
              
        Arguments:
            REQUIRED:
                s: sound pressssure waveform
                sample_rate: sampling rate for s in Hz
                spec_sample_rate: sampling rate for the output spectrogram in Hz. This variable sets sthe overlap for the windows in the STFFT.
                freq_spacing: the time-frequency scale for the spectrogram in Hz. This variable determines the width of the gaussian window. 
            
            OPTIONAL            
                complex = False: returns the absolute value
                use min_freq and max_freq to save space
                nstd = number of standard deviations of the gaussian in one window.
    """
     
    # We need units here!!
    increment = 1.0 / spec_sample_rate
    window_length = nstd / (2.0*np.pi*freq_spacing)
    t,freq,timefreq,rms = gaussian_stft(s, sample_rate, window_length, increment, nstd=nstd, min_freq=min_freq, max_freq=max_freq)

    # rms = spec.std(axis=0, ddof=1)
    if cmplx == False:
        timefreq = np.abs(timefreq)
        
    return t, freq, timefreq, rms


def temporal_envelope(s, sample_rate, cutoff_freq=200.0, resample_rate=None):
    """
        Get the temporal envelope from the sound pressure waveform.

        s: the signal
        sample_rate: the sample rate of the signal
        cutoff_freq: the cutoff frequency of the low pass filter used to create the envelope

        Returns the temporal envelope of the signal, with same sample rate or downsampled.
    """

    #rectify
    srect = np.abs(s)
    #low pass filter
    if cutoff_freq is not None:
        srect = lowpass_filter(srect, sample_rate, cutoff_freq, filter_order=4)
        srect[srect < 0] = 0
        
    if resample_rate is not None:
        lensound = len(srect)
        t=(np.array(range(lensound),dtype=float))/sample_rate
        lenresampled = int(round(float(lensound)*resample_rate/sample_rate))
        (srectresampled, tresampled) = resample(srect, lenresampled, t=t, axis=0, window=None)
        return (srectresampled, tresampled)
    else:   
        return srect


def recursive_ls(root_dir, file_pattern):
    """
        Walks through all the files in root_dir and returns every file whose name matches
        the pattern specified by file_pattern.
    """

    matches = list()
    for root, dirnames, filenames in os.walk(root_dir):
      for filename in fnmatch.filter(filenames, file_pattern):
          matches.append(os.path.join(root, filename))
    return matches


def sox_convert_to_mono(file_path):
    """
        Uses Sox (sox.sourceforge.net) to convert a stereo .wav file to mono.
    """

    root_dir,file_name = os.path.split(file_path)

    base_file_name = file_name[:-4]
    output_file_path = os.path.join(root_dir, '%s_mono.wav' % base_file_name)
    cmd = 'sox \"%s\" -c 1 \"%s\"' % (file_path, output_file_path)
    print cmd
    subprocess.call(cmd, shell=True)


def generate_sine_wave(duration, freq, samprate):
    """
        Generate a pure tone at a given frequency and sample rate for a specified duration.
    """

    t = np.arange(0.0, duration, 1.0 / samprate)
    return np.sin(2*np.pi*freq*t)


def generate_simple_stack(duration, fundamental_freq, samprate, num_harmonics=10):
    nsamps = int(duration*samprate)
    s = np.zeros(nsamps, dtype='float')
    ffreq = 0.0
    for n in range(num_harmonics):
        ffreq += fundamental_freq
        s += generate_sine_wave(duration, ffreq, samprate)
    return s


def generate_harmonic_stack(duration, fundamental_freq, samprate, num_harmonics=10, base=2):

    nsamps = int(duration*samprate)
    s = np.zeros(nsamps, dtype='float')
    for n in range(num_harmonics):
        freq = fundamental_freq * base**n
        s += generate_sine_wave(duration, freq, samprate)
    return s


def modulate_wave(s, samprate, freq):

    t = np.arange(len(s), dtype='float') / samprate
    c = np.sin(2*np.pi*t*freq)
    return c*s


def mtfft(spectrogram, df, dt, Norm=False, Log=False):
    """
        Compute the 2d modulation transfer function for a given time frequency slice.
        return temporal_freq,spectral_freq,mps_pow,mps_phase
    """
    #normalize and mean center the spectrogram 
    sdata = copy.copy(spectrogram)
    if Norm:
        sdata /= sdata.max()
        sdata -= sdata.mean()

    #take the 2D FFT and center it
    smps = fft2(sdata)
    smps = fftshift(smps)

    #compute the log amplitude
    mps_pow = np.abs(smps)**2
    if Log:
        mps_pow = 10*np.log10(mps_pow)

    #compute the phase
    mps_phase = np.angle(smps)

    #compute the axes
    nf = mps_pow.shape[0]
    nt = mps_pow.shape[1]

    spectral_freq = fftshift(fftfreq(nf, d=df[1]-df[0]))
    temporal_freq = fftshift(fftfreq(nt, d=dt[1]-dt[0]))

    """
    nb = sdata.shape[1]
    dwf = np.zeros(nb)
    for ib in range(int(np.ceil((nb+1)/2.0))+1):
        posindx = ib
        negindx = nb-ib+2
        print 'ib=%d, posindx=%d, negindx=%d'% (ib, posindx, negindx )
        dwf[ib]= (ib-1)*(1.0/(df*nb))
        if ib > 1:
            dwf[negindx] =- dwf[ib]

    nt = sdata.shape[0]
    dwt = np.zeros(nt)
    for it in range(0, int(np.ceil((nt+1)/2.0))+1):
        posindx = it
        negindx = nt-it+2
        print 'it=%d, posindx=%d, negindx=%d' % (it, posindx, negindx)
        dwt[it] = (it-1)*(1.0/(nt*dt))
        if it > 1 :
            dwt[negindx] = -dwt[it]

    spectral_freq = dwf
    temporal_freq = dwt
    """

    return spectral_freq, temporal_freq, mps_pow, mps_phase

def mps(spectrogram, df, dt, window=None, Norm=True):
    """
    Calculates the modulation power spectrum using overlapp and add method with a gaussian window of length window in s
    """
    # Check the size of the spectrogram vs dt
    nt = dt.size
    nf = df.size
    if spectrogram.shape[1] != nt and spectrogram.shape[0] != nf:   
        print 'Error in mps. Expected  %d bands in frequency and %d points in time' % (nf, nt)
        print 'Spectrogram had shape %d, %d' % spectrogram.shape
        return 0, 0, 0
        
    # Z-score the flattened spectrogram
    if Norm:
        spectrogram -= spectrogram.mean()
        spectrogram /= spectrogram.std()
        
    if window == None:
        window = dt[-1]/10.0
            
    # Find the number of spectrogram points in the gaussian window 
    if dt[-1] < window:
        print 'Warning in mps: window size is smaller or equal to spectrogram temporal extent.'
        print 'mps will be calculate with a single window'
        nWindow = nt - 1
    else:
        nWindow = mlab.find(dt>= window)[0]
    if nWindow%2 == 0:
        nWindow += 1  # Make it odd size so that we have a symmetric window
        
    if nWindow < 64:
        print 'Error in mps: window size %d pts (%.3.f s) is two small for reasonable estimates' % (nWindow, window)
        return 0, 0, 0
        
    # Generate the Gaussian window
    gt, w = gaussian_window(nWindow, 6)
    tShift = int(gt[-1]/3)
    nchunks = 0
    
    for tmid in range(tShift, nt, tShift):
        
        # No zero padding at this point this could be better
        tstart = tmid-(nWindow-1)/2-1
        if tstart < 0:
            continue
                       
        tend = tmid+(nWindow-1)/2
        if tend > nt:
            break
        nchunks += 1
        
        # Multiply the spectrogram by the window
        wSpect = spectrogram[:,tstart:tend]
        for fInd in range(nf):
            wSpect[fInd,:] = wSpect[fInd,:]*w
            
        # Get the 2d FFT
        wf, wt, mps_pow,mps_phase = mtfft(wSpect, df, dt[tstart:tend])
        if nchunks == 1:
            mps_powAvg = mps_pow
        else:
            mps_powAvg += mps_pow
            
    mps_powAvg /= nchunks
    
    return wf, wt, mps_powAvg


def plot_mps(spectral_freq, temporal_freq, amp, phase=None):

    plt.figure()

    #plot the amplitude
    if phase:
        plt.subplot(2, 1, 1)
        
    #ex = (spectral_freq.min(), spectral_freq.max(), temporal_freq.min(), temporal_freq.max())
    ex = (temporal_freq.min(), temporal_freq.max(), spectral_freq.min()*1e3, spectral_freq.max()*1e3)
    plt.imshow(amp, interpolation='nearest', aspect='auto', origin='lower', cmap=cmap.jet, extent=ex)
    plt.ylabel('Spectral Frequency (Cycles/KHz)')
    plt.xlabel('Temporal Frequency (Hz)')
    plt.colorbar()
    plt.ylim((0,spectral_freq.max()*1e3))
    plt.title('Power')

    #plot the phase
    if phase:
        plt.subplot(2, 1, 2)
        plt.imshow(phase, interpolation='nearest', aspect='auto', origin='lower', cmap=cmap.jet, extent=ex)
        plt.ylabel('Spectral Frequency (Cycles/KHz)')
        plt.xlabel('Temporal Frequency (Hz)')
        plt.ylim((0,spectral_freq.max()*1e3))
        plt.title('Phase')
        plt.colorbar()

def synSpect(b, x):
# Generates a model spectrum made out of gaussian peaks
# fund, sigma, pkmax, dbfloor
# global fundGlobal maxFund minFund

    npeaks = np.size(b)-1  # First element of b is the sampling rate
# amp = 25      # Force 25 dB peaks
    sdpk = 60    # Force 80 hz width

    synS = np.zeros(len(x))


    for i in range(npeaks):
        a = b[i+1]   # To inforce positive peaks only
        synS = synS + a*np.exp(-(x-b[0]*(i+1))**2/(2*sdpk**2))
    
    #if (sum(isinf(synS)) + sum(isnan(synS))):
    #    for i in range(npeaks):
    #        fprintf(1,'%f ', exp(b(i+1)))    
    return synS

    
def residualSyn(vars, x, realS):
    b = vars
    synS = synSpect(b, x)
    
    return realS-synS
    
    #if (sum(isinf(synS)) + sum(isnan(synS)))
    #    for i=1:npeaks
    #        fprintf(1,'%f ', exp(b(i+1)))   


def fundEstimator(soundIn, fs, t=None, debugFig = 0, maxFund = 1500, minFund = 300, lowFc = 200, highFc = 6000, minSaliency = 0.5):
    """
    Estimates the fundamental frequency of a complex sound.
    soundIn is the sound pressure waveformlog spectrogram.
    fs is the sampling rate
    t is a vector of time values in s at which the fundamental will be estimated.
    The sound must include at least 1024 sample points

    The optional parameter with defaults are
    Some user parameters (should be part of the function at some time)
       debugFig = 0         Set to zero to eliminate figures.
       maxFund = 1500       Maximum fundamental frequency
       minFund = 300        Minimum fundamental frequency
       lowFc = 200          Low frequency cut-off for band-passing the signal prior to auto-correlation.
       highFc = 6000        High frequency cut-off
       minSaliency = 0.5    Threshold in the auto-correlation for minimum saliency - returns NaN for pitch values is saliency is below this number

    Returns
           sal     - the time varying pitch saliency - a number between 0 and 1 corresponding to relative size of the first auto-correlation peak
           fund     - the time-varying fundamental in Hz at the same resolution as the spectrogram.
           fund2   - a second peak in the spectrum - not a multiple of the fundamental a sign of a second voice
           form1   - the first formant, if it exists
           form2   - the second formant, if it exists
           form3   - the third formant, if it exists
           soundLen - length of sal, fund, fund2, form1, form2, form3
    """

    # Band-pass filtering signal prior to auto-correlation
    soundLen = len(soundIn)
    nfilt = 1024
    if soundLen < 1024:
        print 'Error in fundEstimator: sound too short for bandpass filtering, len(soundIn)=%d' % soundLen
        return (0, 0, 0, 0, 0, 0, 0)

    # high pass filter the signal
    highpassFilter = firwin(nfilt-1, 2*lowFc/fs, pass_zero=False)
    padlen = min(soundLen-10, 3*len(highpassFilter))
    soundIn = filtfilt(highpassFilter, [1.0], soundIn, padlen=padlen)

    # low pass filter the signal
    lowpassFilter = firwin(nfilt, 2*highFc/fs)
    padlen = min(soundLen-10, 3*len(lowpassFilter))
    soundIn = filtfilt(lowpassFilter, [1.0], soundIn, padlen=padlen)

    # Plot a spectrogram?
    #if debugFig:
    #    plt.figure(9)
    #    (tDebug ,freqDebug ,specDebug , rms) = spectrogram(soundIn, fs, 1000.0, 50, min_freq=0, max_freq=10000, nstd=6, log=True, noise_level_db=50, rectify=True) 
    #    plot_spectrogram(tDebug, freqDebug, specDebug)

    # Initializations and useful variables
    soundLen = len(soundIn)
    sound_dur = soundLen / fs
    
    if t is None:
        # initialize t to be spaced by  1 ms increments if not specified     
        _si = 1e-3
        npts = int(sound_dur / _si)
        t = np.arange(npts) * _si

    nt=len(t)
    soundRMS = np.zeros(nt)
    fund = np.zeros(nt)
    fund2 = np.zeros(nt)
    sal = np.zeros(nt)
    form1 = np.zeros(nt)
    form2 = np.zeros(nt)
    form3 = np.zeros(nt)

    #  Calculate the size of the window for the auto-correlation
    alpha = 5                          # Number of sd in the Gaussian window
    winLen = int(np.fix((2.0*alpha/minFund)*fs))  # Length of Gaussian window based on minFund
    if (winLen%2 == 0):  # Make a symmetric window
        winLen += 1
        
    winLen2 = 2**12+1   # This looks like a good size for LPC - 4097 points

    gt, w = gaussian_window(winLen, alpha)
    gt2, w2 = gaussian_window(winLen2, alpha)
    maxlags = int(2*ceil((float(fs)/minFund)))

    # First calculate the rms in each window
    for it in range(nt):
        tval = t[it]               # Center of window in time
        if tval >= sound_dur:
            continue
        tind = int(np.fix(tval*fs))    # Center of window in ind
        tstart = tind - (winLen-1)/2
        tend = tind + (winLen-1)/2
    
        if tstart < 0:
            winstart = - tstart
            tstart = 0
        else:
            winstart = 0
        
        if tend >= soundLen:
            windend = winLen - (tend-soundLen+1) - 1
            tend = soundLen-1
        else:
            windend = winLen-1
            
        soundWin = soundIn[tstart:tend]*w[winstart:windend]
        soundRMS[it] = np.std(soundWin)
    
    soundRMSMax = max(soundRMS)

    # Calculate the auto-correlation in windowed segments and obtain 4 guess values of the fundamental
    # fundCorrGuess - guess from the auto-correlation function
    # fundCorrAmpGuess - guess form the amplitude of the auto-correlation function
    # fundCepGuess - guess from the cepstrum
    # fundStackGuess - guess taken from a fit of the power spectrum with a harmonic stack, using the fundCepGuess as a starting point
    #  Current version use fundStackGuess as the best estimate...

    soundlen = 0
    for it in range(nt):
        fund[it] = float('nan')
        sal[it] = float('nan')
        fund2[it] = float('nan')
        form1[it] = float('nan')
        form2[it] = float('nan')
        form3[it] = float('nan')
        
        if (soundRMS[it] < soundRMSMax*0.1):
            continue
    
        soundlen += 1
        tval = t[it]               # Center of window in time
        if tval >= sound_dur:       # This should not happen here because the RMS should be zero
            continue
        tind = int(np.fix(tval*fs))    # Center of window in ind
        tstart = tind - (winLen-1)/2
        tend = tind + (winLen-1)/2
    
        if tstart < 0:
            winstart = - tstart
            tstart = 0
        else:
            winstart = 0
        
        if tend >= soundLen:
            windend = winLen - (tend-soundLen+1) - 1
            tend = soundLen-1
        else:
            windend = winLen-1
            
        tstart2 = tind - (winLen2-1)/2
        tend2 = tind + (winLen2-1)/2
    
        if tstart2 < 0:
            winstart2 = - tstart2
            tstart2 = 0
        else:
            winstart2 = 0
        
        if tend2 >= soundLen:
            windend2 = winLen2 - (tend2-soundLen+1) - 1
            tend2 = soundLen-1
        else:
            windend2 = winLen2-1
            
        soundWin = soundIn[tstart:tend]*w[winstart:windend]
              
        soundWin2 = soundIn[tstart2:tend2]*w2[winstart2:windend2]
        
        # Apply LPC to get time-varying formants and one additional guess for the fundamental frequency
        import scikits.talkbox as talkbox
        A, E, K = talkbox.lpc(soundWin2, 8)    # 8 degree polynomial
        rts = np.roots(A)          # Find the roots of A
        rts = rts[np.imag(rts)>=0]  # Keep only half of them
        angz = np.arctan2(np.imag(rts),np.real(rts))
    
        # Calculate the frequencies and the bandwidth of the formants
        frqsFormants = angz*(fs/(2*np.pi))
        indices = np.argsort(frqsFormants)
        bw = -1/2*(fs/(2*np.pi))*np.log(np.abs(rts))
    
        # Keep formants above 500 Hz and with bandwidth < 500 # This was 1000 for bird calls
        formants = []
        for kk in indices:
            if ( frqsFormants[kk]>500 and bw[kk] < 500):        
                formants.append(frqsFormants[kk])
        formants = np.array(formants) 
        
        if len(formants) > 0 : 
            form1[it] = formants[0]
        if len(formants) > 1 : 
            form2[it] = formants[1]
        if len(formants) > 2 : 
            form3[it] = formants[2]

        # Calculate the auto-correlation
        lags = np.arange(-maxlags, maxlags+1, 1)
        autoCorr = correlation_function(soundWin, soundWin, lags)
        ind0 = int(mlab.find(lags == 0))  # need to find lag zero index
    
        # find peaks
        indPeaksCorr = detect_peaks(autoCorr, mph=max(autoCorr)/10)
    
        # Eliminate center peak and all peaks too close to middle    
        indPeaksCorr = np.delete(indPeaksCorr,mlab.find( (indPeaksCorr-ind0) < fs/maxFund))
        pksCorr = autoCorr[indPeaksCorr]
    
        # Find max peak
        if len(pksCorr)==0:
            pitchSaliency = 0.1               # 0.1 goes with the detection of peaks greater than max/10
        else:
            indIndMax = mlab.find(pksCorr == max(pksCorr))[0]
            indMax = indPeaksCorr[indIndMax]   
            fundCorrGuess = fs/abs(lags[indMax])
            pitchSaliency = autoCorr[indMax]/autoCorr[ind0]

        sal[it] = pitchSaliency
    
        if sal[it] < minSaliency:
            continue

        # Calculate the envelope of the auto-correlation after rectification
        envCorr = temporal_envelope(autoCorr, fs, cutoff_freq=maxFund, resample_rate=None) 
        locsEnvCorr = detect_peaks(envCorr, mph=max(envCorr)/10)
        pksEnvCorr = envCorr[locsEnvCorr]
    
        # The max peak should be around zero
        indIndEnvMax = mlab.find(pksEnvCorr == max(pksEnvCorr))
        
        if indIndEnvMax.size > 1:           # Two identical peaks? Take the first one
            indIndEnvMax = indIndEnvMax[0]
          
        # Take the first peak not in the middle
        if indIndEnvMax+2 > len(locsEnvCorr):
            fundCorrAmpGuess = fundCorrGuess
            indEnvMax = indMax
        else:
            indEnvMax = locsEnvCorr[indIndEnvMax+1]
            fundCorrAmpGuess = fs/lags[indEnvMax]

        # Calculate power spectrum and cepstrum
        Y = fft(soundWin, n=winLen+1)
        f = (fs/2.0)*(np.array(range((winLen+1)/2+1), dtype=float)/float((winLen+1)/2))
        fhigh = mlab.find(f >= highFc)[0]
    
        powSound = 20.0*np.log10(np.abs(Y[0:(winLen+1)/2+1]))    # This is the power spectrum
        powSoundGood = powSound[0:fhigh]
        maxPow = max(powSoundGood)
        powSoundGood = powSoundGood - maxPow   # Set zero as the peak amplitude
        powSoundGood[powSoundGood < - 60] = -60    
    
        # Calculate coarse spectral enveloppe
        p = np.polyfit(f[0:fhigh], powSoundGood, 3)
        powAmp = np.polyval(p, f[0:fhigh]) 
    
        # Cepstrum
        CY = dct(powSoundGood-powAmp, norm = 'ortho')            
    
        tCY = 1000.0*np.array(range(len(CY)))/fs          # Units of Cepstrum in ms
        fCY = 1000.0/tCY # Corresponding fundamental frequency in Hz.
        lowInd = mlab.find(fCY<lowFc)
        if lowInd.size > 0:
            flowCY = mlab.find(fCY < lowFc)[0]
        else:
            flowCY = fCY.size
            
        fhighCY = mlab.find(fCY < highFc)[0]
    
        # Find peak of Cepstrum
        indPk = mlab.find(CY[fhighCY:flowCY] == max(CY[fhighCY:flowCY]))[-1]
        indPk = fhighCY + indPk 
        fmass = 0
        mass = 0
        indTry = indPk
        while (CY[indTry] > 0):
            fmass = fmass + fCY[indTry]*CY[indTry]
            mass = mass + CY[indTry]
            indTry = indTry + 1
            if indTry >= len(CY):
                break

        indTry = indPk - 1
        if (indTry >= 0 ):
            while (CY[indTry] > 0):
                fmass = fmass + fCY[indTry]*CY[indTry]
                mass = mass + CY[indTry]
                indTry = indTry - 1
                if indTry < 0:
                    break

        fGuess = fmass/mass
    
        if (fGuess == 0  or np.isnan(fGuess) or np.isinf(fGuess) ):              # Failure of cepstral method
            fGuess = fundCorrGuess

        fundCepGuess = fGuess
    
        # Force fundamendal to be bounded
        if (fundCepGuess > maxFund ):
            i = 2
            while(fundCepGuess > maxFund):
                fundCepGuess = fGuess/i
                i += 1
        elif (fundCepGuess < minFund):
            i = 2
            while(fundCepGuess < minFund):
                fundCepGuess = fGuess*i
                i += 1
    
        # Fit Gaussian harmonic stack
        maxPow = max(powSoundGood-powAmp)

        # This is the matlab code...
        # fundFitCep = NonLinearModel.fit(f(1:fhigh)', powSoundGood'-powAmp, @synSpect, [fundCepGuess ones(1,9).*log(maxPow)])
        # modelPowCep = synSpect(double(fundFitCep.Coefficients(:,1)), f(1:fhigh))

        vars = np.concatenate(([fundCorrGuess], np.ones(9)*np.log(maxPow)))
        bout = leastsq(residualSyn, vars, args = (f[0:fhigh], powSoundGood-powAmp)) 
        modelPowCep = synSpect(bout[0], f[0:fhigh])
        errCep = sum((powSoundGood - powAmp - modelPowCep)**2)
    
        vars = np.concatenate(([fundCorrGuess*2], np.ones(9)*np.log(maxPow)))
        bout2 = leastsq(residualSyn, vars, args = (f[0:fhigh], powSoundGood-powAmp)) 
        modelPowCep2 = synSpect(bout2[0], f[0:fhigh])
        errCep2 = sum((powSoundGood - powAmp - modelPowCep2)**2)
    
        if errCep2 < errCep:
            bout = bout2
            modelPowCep =  modelPowCep2

        fundStackGuess = bout[0][0]
        if (fundStackGuess > maxFund) or (fundStackGuess < minFund ):
            fundStackGuess = float('nan')

    
        # A second cepstrum for the second voice
        #     CY2 = dct(powSoundGood-powAmp'- modelPowCep)
                
        fund[it] = fundStackGuess        
    
        if  not np.isnan(fundStackGuess):
            powLeft = powSoundGood- powAmp - modelPowCep
            maxPow2 = max(powLeft)
            f2 = 0
            if ( maxPow2 > maxPow*0.5):    # Possible second peak in central area as indicator of second voice.
                f2 = f[mlab.find(powLeft == maxPow2)]
                if ( f2 > 1000 and f2 < 4000):
                    if (pitchSaliency > minSaliency):
                        fund2[it] = f2

#%     modelPowCorrAmp = synSpect(double(fundFitCorrAmp.Coefficients(:,1)), f(1:fhigh))
#%     
#%     errCorr = sum((powSoundGood - powAmp' - modelPowCorr).^2)
#%     errCorrAmp = sum((powSoundGood - powAmp' - modelPowCorrAmp).^2)
#%     errCorrSum = sum((powSoundGood - powAmp' - (modelPowCorr+modelPowCorrAmp) ).^2)
#%       
#%     f1 = double(fundFitCorr.Coefficients(1,1))
#%     f2 = double(fundFitCorrAmp.Coefficients(1,1))
#%     
#%     if (pitchSaliency > minSaliency)
#%         if (errCorr < errCorrAmp)
#%             fund(it) = f1
#%             if errCorrSum < errCorr
#%                 fund2(it) = f2
#%             end
#%         else
#%             fund(it) = f2
#%             if errCorrSum < errCorrAmp
#%                 fund2(it) = f1
#%             end
#%         end
#%         
#%     end

        if (debugFig ):
            plt.figure(10)
            plt.subplot(4,1,1)
            plt.cla()
            plt.plot(soundWin)
#         f1 = double(fundFitCorr.Coefficients(1,1))
#         f2 = double(fundFitCorrAmp.Coefficients(1,1))
            titleStr = 'Saliency = %.2f F0 AC = %.2f ACA = %.2f Cep = %.2f St = %.2f(Hz)' % (pitchSaliency, fundCorrGuess, fundCorrAmpGuess, fundCepGuess, fundStackGuess)
            plt.title(titleStr)
        
            plt.subplot(4,1,2)
            plt.cla()
            plt.plot(1000*(lags/fs), autoCorr)
            plt.plot([1000.*lags[indMax]/fs, 1000*lags[indMax]/fs], [0, autoCorr[ind0]], 'k')
            plt.plot(1000.*lags/fs, envCorr, 'r', linewidth= 2)
            plt.plot([1000*lags[indEnvMax]/fs, 1000*lags[indEnvMax]/fs], [0, autoCorr[ind0]], 'g')
            plt.xlabel('Time (ms)')
              
            plt.subplot(4,1,3)
            plt.cla()
            plt.plot(f[0:fhigh],powSoundGood)
            plt.axis([0, highFc, -60, 0])
            plt.plot(f[0:fhigh], powAmp, 'b--')
            plt.plot(f[0:fhigh], modelPowCep + powAmp, 'k')
            # plt.plot(f(1:fhigh), modelPowCorrAmp + powAmp', 'g')
        
            for ih in range(1,6):
                plt.plot([fundCorrGuess*ih, fundCorrGuess*ih], [-60, 0], 'r')
                plt.plot([fundStackGuess*ih, fundStackGuess*ih], [-60, 0], 'k')
                plt.plot([fundCepGuess*ih, fundCepGuess*ih], [-60, 0], 'y')

            if f2 != 0: 
                plt.plot([f2, f2], [-60, 0], 'g')

            plt.xlabel('Frequency (Hz)')
            # title(sprintf('Err1 = %.1f Err2 = %.1f', errCorr, errCorrAmp))
        
            plt.subplot(4,1,4)
            plt.cla()
            plt.plot(tCY, CY)
#         plot(tCY, CY2, 'k--')
            plt.plot([1000/fundCorrGuess, 1000/fundCorrGuess], [0, max(CY)], 'r')
            plt.plot([1000/fundStackGuess, 1000/fundStackGuess], [0, max(CY)], 'k')
            plt.plot([1000/fundCepGuess, 1000/fundCepGuess], [0, max(CY)], 'k')
        
            #%         plot([(pkClosest-1)/fs (pkClosest-1)/fs], [0 max(CY)], 'g')
            #%         if ~isempty(ipk2)
            #%             plot([(pk2-1)/fs (pk2-1)/fs], [0 max(CY)], 'b')
            #%         end
            #%         for ip=1:length(pks)
            #%             plot([(locs(ip)-1)/fs (locs(ip)-1)/fs], [0 pks(ip)/4], 'r')
            #%         end
            plt.axis([0, 1000*np.size(CY)/(2*fs), 0, max(CY)])
            plt.xlabel('Time (ms)')

            plt.pause(1)
    
    # Fix formants.
    meanf1 = np.mean(form1[~np.isnan(form1)])
    meanf2 = np.mean(form2[~np.isnan(form2)])
    meanf3 = np.mean(form3[~np.isnan(form3)])

    for it in range(nt):
        if ~np.isnan(form1[it]):
            df11 = np.abs(form1[it]-meanf1)
            df12 = np.abs(form1[it]-meanf2)
            df13 = np.abs(form1[it]-meanf3)
            if df12 < df11:
                if df13 < df12:
                    if ~np.isnan(form3[it]):
                        df33 = np.abs(form3[it]-meanf3)
                        if df13 < df33:
                            form3[it] = form1[it]
                    else:
                      form3[it] = form1[it]
                else:
                    if ~np.isnan(form2[it]):
                        df22 = np.abs(form2[it]-meanf2)
                        if df12 < df22:
                            form2[it] = form1[it]
                    else:
                        form2[it] = form1[it]
                form1[it] = float('nan')
            if ~np.isnan(form2[it]):  
                df21 = np.abs(form2[it]-meanf1)
                df22 = np.abs(form2[it]-meanf2)
                df23 = np.abs(form2[it]-meanf3)
                if df21 < df22 :
                    if ~np.isnan(form1[it]):
                        df11 = np.abs(form1[it]-meanf1)
                        if df21 < df11:
                            form1[it] = form2[it]
                    else:
                      form1[it] = form2[it]
                    form2[it] = float('nan')
                elif df23 < df22:
                    if ~np.isnan(form3[it]):
                        df33 = np.abs(form3[it]-meanf3)
                        if df23 < df33:
                            form3[it] = form2[it]
                    else:
                        form3[it] = form2[it]
                    form2[it] = float('nan')
            if ~np.isnan(form3[it]):
                df31 = np.abs(form3[it]-meanf1)
                df32 = np.abs(form3[it]-meanf2)
                df33 = np.abs(form3[it]-meanf3)
                if df32 < df33:
                    if df31 < df32:
                        if ~np.isnan(form1[it]):
                            df11 = np.abs(form1[it]-meanf1)
                            if df31 < df11:
                                form1[it] = form3[it]
                        else:
                            form1[it] = form3[it]
                    else:
                        if ~np.isnan(form2[it]):
                            df22 = np.abs(form2[it]-meanf2)
                            if df32 < df22:
                                form2[it] = form3[it]
                        else:
                            form2[it] = form3[it]
                    form3[it] = float('nan')

    return (sal, fund, fund2, form1, form2, form3, soundlen)



def get_mps(t, freq, spec):
    "Computes the MPS of a spectrogram (idealy a log-spectrogram) or other REAL time-freq representation"
    mps = fftshift(fft2(spec))
    amps = np.real(mps * np.conj(mps))
    nf = mps.shape[0]
    nt = mps.shape[1]
    wfreq = fftshift(fftfreq(nf, d=freq[1] - freq[0]))
    wt = fftshift(fftfreq(nt, d=t[1] - t[0]))
    return wt, wfreq, mps, amps

def inverse_mps(mps):
    "Inverts a MPS back to a spectrogram"
    spec = ifft2(ifftshift(mps))
    return spec



def play_signal(s, normalize = False):
    "quick and easy temporary play"
    wf = WavFile()
    wf.sample_rate = 44100 #standard samp rate
    wf.data = s
    wf.to_wav("/tmp/README.wav", normalize)
    play_sound("/tmp/README.wav")



def inverse_spectrogram(spec, s_len,
    sample_rate, spec_sample_rate, freq_spacing, min_freq=0, max_freq=None, nstd=6, log=True, noise_level_db=80, rectify=True):
    """turns the complex spectrogram into a signal

    inverts by repeating the process on a string-of-ones
    """

    spec_copy = spec.copy()
    if log:
        spec_copy = 10**(spec_copy)
    spec_tranpose = spec.transpose() # spec_tranpose[time][frequency]

    hnwinlen = len(spec) - 1
    nincrement = int(np.round(float(sample_rate)/spec_sample_rate))

    gauss_t = np.arange(-hnwinlen, hnwinlen+1, 1.0)
    gauss_std = float(2*hnwinlen) / float(nstd)
    gauss_window = np.exp(-gauss_t**2 / (2.0*gauss_std**2)) / (gauss_std*np.sqrt(2*np.pi))
    
    s = np.zeros(s_len + 2*hnwinlen+1)
    w = np.zeros(s_len + 2*hnwinlen+1)

    for i in range(len(spec_tranpose)):
        sample = i * nincrement
        spec_slice = np.concatenate((spec_tranpose[i][:0:-1].conj(), spec_tranpose[i]))
        s[sample:sample+2*hnwinlen+1] += gauss_window * ifft(ifftshift(spec_slice))
        w[sample:sample+2*hnwinlen+1] += gauss_window ** 2
    s /= w
    return s[hnwinlen:hnwinlen+s_len]


def inverse_real_spectrogram(spec, s_len,
    sample_rate, spec_sample_rate, freq_spacing, min_freq=0, max_freq=None, nstd=6, log=True, noise_level_db=80, rectify=True, iterations = 10):
    "inverts a real spectrogram into a signal using the griffith/lim algorithm"
    spec_magnitude = spec.copy()

    if log:
        spec_magnitude = 10**spec_magnitude
    estimated = inverse_spectrogram(spec_magnitude, s_len, sample_rate, spec_sample_rate, freq_spacing, min_freq, max_freq, nstd, log=False)
    for i in range(iterations):
        phase_spec = spectrogram(estimated, sample_rate, spec_sample_rate, freq_spacing, min_freq, max_freq, nstd, log=False)[2]
        error = ((abs(spec_magnitude) - abs(phase_spec))**2).sum() / (abs(spec_magnitude)**2).sum()
        print 'the error after iteration %d is %f' % (i+i, error)
        spec_angle = np.angle(phase_spec)
        estimated_spec = spec_magnitude * np.exp(1j*spec_angle)
        estimated = inverse_spectrogram(estimated_spec, s_len, sample_rate, spec_sample_rate, freq_spacing, min_freq, max_freq, nstd, log=False)
    return estimated


def log_transform(x, dbnoise=100, normalize=False):
    """ Takes the log of a power spectrum or spectrogram to convert into decibels.

    :param x: The power spectrum or spectrogram. The contents of x will be replaced with the log version.
    :param dbnoise: The noise level in decibels. Anything lower than dbnoise will be set to zero.
    """
    x /= x.max()
    zi = x > 0
    x[zi] = 20*np.log10(x[zi]) + dbnoise
    x[x < 0] = 0
    if normalize:
        x /= x.max()


def spec_stats(spec_t, spec_freq, spec):
    """ Compute time-varying statistics on a spectrogram (or log spectrogram).

    :param spec_t: Spectrogram times with shape (num_time_points)
    :param spec_freq: Spectrogram frequencies with shape (num_freq)
    :param spec: Spectrogram of shape (num_freq, num_time_points)

    :return:
    """

    # normalize each time point by it's sum to create a probability distribution
    nfreq,nt = spec.shape
    spec_p = deepcopy(spec)
    spec_p -= spec_p.min()
    spec_p_sum = spec_p.sum(axis=0)
    spec_p /= spec_p_sum

    # compute mean frequency
    freq_mean = np.dot(spec_p.T, spec_freq)

    # compute quantiles
    spec_p_csum = np.cumsum(spec_p, axis=0)

    qvals = [0.25, 0.5, 0.75]
    Q = np.zeros([len(qvals), nt])
    for t in range(nt):
        for k,q in enumerate(qvals):
            i = spec_p_csum[:, t] <= q
            if i.sum() > 0:
                fi = np.max(np.where(i)[0])
                Q[k, t] = spec_freq[fi]

    stats = dict()
    stats['Q'] = Q
    stats['qvals'] = qvals
    stats['freq_mean'] = freq_mean

    return stats

