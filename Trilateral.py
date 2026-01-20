import nidaqmx
import numpy as np

from scipy.fft import fft, fftfreq,ifft
from scipy.signal import find_peaks
import csv
from nidaqmx.constants import AcquisitionType, READ_ALL_AVAILABLE
import matplotlib.pyplot as plt

def maxfreq():
    with nidaqmx.Task() as task:

        task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai1")
        frmax=[]
        
        s_hz=60000
        s_n=240000
        
        task.timing.cfg_samp_clk_timing(s_hz, sample_mode=AcquisitionType.FINITE, samps_per_chan=s_n) 
        data = task.read(READ_ALL_AVAILABLE)
           
        fft_result =fft(data)
        fft_magnitude = np.abs(fft_result)
        freqs = fftfreq(len(data), d=1/(1*s_hz))

        half_n = len(data) // 2
        fft_magnitude = fft_magnitude[:half_n]
        freqs = freqs[:half_n] 

        for i in range (10001): #low pass filter
            fft_magnitude[i]=0
            
        '''
        plt.plot(freqs, fft_magnitude)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('FFT Spectrum')
        plt.show()
        plt.close()'''
        
        '''plt.plot(data)
        plt.title('Time-domain Signal')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.show()
        plt.close()'''

        
        
        max_index = np.argmax(fft_magnitude)
        dominant_freq = freqs[max_index]
        frmax.append(dominant_freq)

        return dominant_freq
    
def acsig(fs=60000, samples=240000):
 
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai0:3")  
            task.timing.cfg_samp_clk_timing(rate=fs, samps_per_chan=samples)
            data = task.read(number_of_samples_per_channel=samples)
        return np.array(data)

def tdelay(sig, ref, fs):
    """Time delay estimation using GCC-PHAT."""
    n = len(sig) + len(ref)

    # FFT of both signals
    SIG = fft(sig, n=n)
    REF = fft(ref, n=n)

    # Cross-power spectrum
    R = SIG * np.conj(REF)

    # Apply PHAT weighting (normalize magnitude)
    R /= np.abs(R) + 1e-10  # avoid division by zero

    # Inverse FFT to obtain cross-correlation
    corr = np.real(ifft(R))

    # Rearrange because correlation is circular
    corr = np.concatenate((corr[-(len(ref)-1):], corr[:len(sig)]))

    # Find lag (in samples)
    lag = np.argmax(corr) - (len(ref) - 1)

    # Convert lag to time (seconds)
    return lag / fs


def doa(signals, fs, positions, c=1500.0):
    ref = signals[0]
    delays = [0.0]
    for i in range(1, signals.shape[0]):
        delays.append(tdelay(signals[i], ref, fs))
    delays = np.array(delays)

    r_ref = positions[0]
    A = positions[1:] - r_ref
    b = c * delays[1:]

    u, *_ = np.linalg.lstsq(A, b, rcond=None)
    u = u / np.linalg.norm(u)

    az = np.degrees(np.arctan2(u[1], u[0]))
    el = np.degrees(np.arcsin(u[2]))
    return az, el, u

def estimate_distance(data, fs, sl_db, sensitivity_db=-180, alpha=0.0):
    # Convert volts to uPa using hydrophone sensitivity
    # sensitivity_db = 20 * log10(V/µPa)  -> V = uPa * 10^(sens/20)
    calib_factor = 10**(-sensitivity_db / 20)  # uPa per Volt
    pressure = data * calib_factor

    # Compute RMS pressure
    p_rms = np.sqrt(np.mean(pressure**2))

    # Reference pressure in water: 1 uPa
    rl_db = 20 * np.log10(p_rms / 1.0)

    tl = sl_db - rl_db
    r = 10**(tl / 20.0)

    if alpha > 0:
        # in case of absorbtion
        for _ in range(10):
            r = 10**((tl - alpha * r) / 20.0)

    return r, rl_db

if __name__ == "__main__":
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0],
        [0.0, 0.05, 0.0],
        [0.0, 0.0, 0.05]
    ])

    fm = maxfreq()
    fs=60000.0
    samples = 240000
    signals = acsig(fs, samples)

    az, el, u = doa(signals, fs, positions)
    print(f"Azimuth: {az:.2f}°, Elevation: {el:.2f}°")

    
    data = signals[0]
    sensitivity_db=-180 # hydrophone sensitivity in db re upa
    sl_db = 180.0 #pinger decibels 

    r, rl_db = estimate_distance(data, fs, sl_db)
    print(f"Estimated distance: {r:.2f} m")
