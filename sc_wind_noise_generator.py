"""
Single-Channel Wind Noise Generator

Authors   : Daniele Mirabilii and Emanuël Habets

Reference : D. Mirabilii, A. Lodermeyer, F. Czwielong, S. Becker and E.A P. Habets, 
            Simulating wind noise with airflow speed-dependent characteristics, 
            Proc. of International Workshop on Acoustic Signal Enhancement (IWAENC), 2022.

Copyright (C) 2023 Friedrich-Alexander-Universität Erlangen-Nürnberg, Germany

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import time
import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
import spectrum
import soundfile as sf
import sounddevice as sd

class WindNoiseGenerator:
    """Wind Noise Generator Class"""

    def __init__(self, fs=48000, duration=5, generate=True, wind_profile=None, gustiness=3, short_term_var=True, start_seed=None):
        """Initizalize object"""

        self.fs = fs
        self.duration = duration
        self.samples = fs * duration
        self.generate = generate
        self.gustiness = gustiness
        self.wind_profile = wind_profile
        self.short_term_var = short_term_var
        if start_seed is not None:
            np.random.seed(start_seed)

    def generate_wind_noise(self):
        """Generate single-channel wind noise by filtering excitation signal"""

        if self.generate:
            wind_profile = self._generate_wind_speed_profile()
        else:
            wind_profile = self._import_wind_speed_profile()

        exc = self.generate_excitation_signal(wind_profile)
        exc_filtered = self._filter(exc, wind_profile, 2048)
        exc_filtered = 0.95*exc_filtered / \
            max(np.abs(exc_filtered))

        return exc_filtered, wind_profile

    def generate_excitation_signal(self, wind_profile):
        """Generate excitation signal"""

        window_size = 128
        hops = window_size // 2  # overlap
        hann_window = np.hanning(window_size)  # hanning window

        wgn = np.concatenate(
            (np.zeros(window_size), np.random.randn(self.samples), np.zeros(window_size)))
        wgn_length = len(wgn)

        lt_var = self._generate_long_term_variance(wind_profile)
        lt_var = np.concatenate((np.zeros(window_size), lt_var, np.zeros(window_size)))

        st_var = self._generate_short_term_variance_garch(wind_profile)
        cond_var = np.abs(st_var)

        num_windows = (wgn_length - window_size) // hops + 1
        exc = np.zeros(wgn_length)

        for time_frame in range(num_windows-1):
            start_idx = time_frame * hops
            end_idx = start_idx + window_size
            idx = np.arange(start_idx, end_idx)

            gain_ltst = lt_var[idx]
            if self.short_term_var:
                gain_ltst *= np.sqrt(cond_var[time_frame])
            noise_seg_ltst = gain_ltst * wgn[idx] * hann_window
            exc[idx] += noise_seg_ltst

        exc = exc[window_size:-window_size]

        return exc

    def _generate_short_term_variance_garch(self, wind_profile):
        """Generate short-term variance of GARCH process"""

        window_size = 128
        hops = window_size // 2 # overlap

        profile = np.concatenate(
            (2 * np.ones(window_size), wind_profile, 2 * np.ones(window_size)))
        profile_length = len(profile)

        num_windows = (profile_length - window_size) // hops + 1
        st_var = np.zeros(num_windows)
        cond_var = np.zeros(num_windows)

        for time_frame in range(num_windows):
            start_idx = time_frame * hops
            end_idx = start_idx + window_size
            idx = np.arange(start_idx, end_idx)

            speed = np.clip(np.mean(profile[idx]), 2, 18)
            alpha, beta, omega = self._speed2par(speed)

            if alpha + beta > 1:
                beta = 0

            cond_var[time_frame] = omega + alpha * \
                st_var[time_frame-1]**2 + beta*(cond_var[time_frame-1])
            st_var[time_frame] = np.sqrt(np.abs(cond_var[time_frame])) * \
                np.random.randn()

        return st_var/max(np.abs(st_var))

    def _generate_long_term_variance(self, wind_profile):
        """Generate long-term variance"""

        # Regression parameter noise variance/wind speed
        regression_coeff = np.array([8.00071114414022, -220.332082908370])

        # Long-term noise variance based on wind speed profile in dB scale
        variance_profile_db = np.polyval(regression_coeff, wind_profile)

        # Long-term noise variance in linear scale
        variance_profile = 10 ** (variance_profile_db / 10)
        var_lt = np.sqrt(np.abs(variance_profile))  # long-term gain

        return var_lt

    def _generate_wind_speed_profile(self, b_par=2, a_par=2):
        """Generate the wind speed profile by sampling a Weibull distribution"""

        speed_points = int(
            self.gustiness)  # gustiness, 1 = constant speed, 10 = highly-variable speed

        # Sample from the Weibull distribution (change b and a for different distributions)
        wind_speed_profile_lt = b_par * np.random.weibull(a_par, speed_points)

        # Interpolate speed values as required audio samples
        wind_speed_profile = sp.signal.resample(
            wind_speed_profile_lt, self.samples)

        # Additive speed fluctuations
        fluctuations = 10 * np.random.randn(self.samples)

        # Smoothing of the fluctuations
        hann_window = np.hanning(self.fs * 100e-3)
        hann_window /= sum(hann_window)  # hanning window for the smoothing
        fluctuations = sp.signal.lfilter(hann_window, 1, fluctuations)

        # Add the fluctuations to the generated wind speed profile
        wind_speed_profile += fluctuations

        return wind_speed_profile

    def _import_wind_speed_profile(self):
        """Read the wind speed profile from input"""

        wind_speed_profile_lt = self.wind_profile  # load speed values

        # Interpolate speed values as required audio samples
        wind_speed_profile = sp.signal.resample(
            wind_speed_profile_lt, self.samples)
        fluctuations = 10 * np.random.randn(self.samples)  # additive speed fluctuations

        # Smoothing of the fluctuations
        hann_window = np.hanning(self.fs * 100e-3)
        hann_window /= sum(hann_window)  # hanning window for the smoothing
        fluctuations = sp.signal.lfilter(hann_window, 1, fluctuations)

        # Add the fluctuations to the generated wind speed profile
        wind_speed_profile += fluctuations

        return wind_speed_profile

    def _filter(self, exc, wind_profile, window_size):
        """Filter the excitation signals with the AR filter coefficients"""

        hops = window_size // 2  # overlap
        hann_window = np.hanning(window_size)  # hanning window

        profile = np.concatenate(
            (2 * np.ones(window_size), wind_profile, 2 * np.ones(window_size)))

        exc = np.concatenate((np.zeros(window_size), exc, np.zeros(window_size)))
        exc_length = len(exc)

        # Overlap-add approach for the time-varying filtering of the excitation signal
        num_windows = (exc_length - window_size) // hops + 1
        exc_filtered = np.zeros(exc_length)

        for time_frame in range(num_windows):
            start_idx = time_frame * hops
            end_idx = start_idx + window_size
            idx = np.arange(start_idx, end_idx)

            speed = np.clip(np.mean(profile[idx]), 2, 18)
            lpc = self._lsf2lpc(speed)

            exc_seg = exc[idx] * hann_window
            exc_seg_filtered = sp.signal.lfilter(
                np.array([1.0]), lpc, exc_seg)

            exc_filtered[idx] += exc_seg_filtered

        exc_filtered = exc_filtered[window_size:-window_size]

        return exc_filtered

    def _speed2par(self, speed):
        """Convert speed to GARCH parameters"""

        gp_alpha = np.array([-2.73244444508231e-05, 0.00141129711949206, -
                            0.0274652794467908, 0.257613241095714, -0.139824587447063])
        gp_beta = np.array(
            [-9.75160902595897e-05, 0.00464300106846736, -0.0871968755558256, 0.651013973757802])
        gp_omega = np.array(
            [9.69585296574741e-05, -0.00231853830578967, 0.0124681159197788])

        alpha = np.polyval(gp_alpha, speed)
        beta = np.polyval(gp_beta, speed)
        omega = np.polyval(gp_omega, speed)

        return alpha, beta, omega

    def _lsf2lpc(self, speed):
        """Generate LPC coefficients from the LSF-speed models given a speed value"""

        # Regression coefficients of the LFS-speed model
        # The n-th LFS coefficient corresponds to the n-th column
        regression_coeff = np.array([[-2.63412497797108e-06, 5.93162248595821e-05,
                                      0.000215613938043173, -0.000149723789407121,
                                      -0.000213703084399375],
                                     [9.50240139044154e-05,	-0.00271741166649528,
                                      -0.0103783584000284, 0.00483963669507075,
                                      0.00931864887930701],
                                     [-0.000699199223507821, 0.0428714179385289,
                                      0.177250839818556, -0.0329542145779793,
                                      -0.129910107562929],
                                     [0.0106849674771013, -0.234688122194936,
                                      -1.21337646113093, -0.168053225019258,
                                      0.568371362156217],
                                     [-0.000966851130291645, 0.541693139684727,
                                      3.24796925730457, 2.54984352038733,
                                      1.86097523205089]])
        order = 5

        # Estimate LFS based on the speed value
        lfs_estimated = np.zeros(order)

        for order_idx in range(order):
            lfs_estimated[order_idx] = np.polyval(regression_coeff[:, order_idx], speed)

        # Convert LFS into LPC coefficients
        lpc_a = spectrum.lsf2poly(lfs_estimated)

        return lpc_a

    def plot_signals(self, wns, wind_profile):
        """
        Plot the generated wind noise signals and the associated wind profile.

        Example: 
         wn = WindNoiseGenerator(fs=16000, duration=10)
         wn_sample, wind_profile = wn.generate_wind_noise()
         wn.plot_signals(wn_sample, wind_profile)
        """

        time_ind = np.arange(0, self.duration, 1. / self.fs)
        wns = wns[:time_ind.shape[0]]

        fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)
        axs[0].plot(time_ind, wns)
        axs[0].set_ylabel('Amplitude')
        axs[0].grid(True)
        axs[0].autoscale(enable=True, axis='x', tight=True)
        axs[0].set_ylim(-1, 1)

        axs[2].plot(time_ind, np.abs(wind_profile), label='Wind Profile')
        axs[2].set_ylim(0, 6)
        axs[2].set_ylabel('Wind speed [m/s]')
        axs[2].grid(True)
        axs[2].axis('tight')

        axs[1].specgram(
            wns, Fs=self.fs, NFFT=512, noverlap=128,
            mode='magnitude', scale='dB', cmap='inferno',
            vmin=20 * np.log10(np.max(wns)) - 120
        )
        axs[1].autoscale(enable=True, axis='x', tight=True)
        axs[1].set_ylabel('Frequency [Hz]')
        axs[1].set_ylim(0, 8000)
        axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x/1e3:g}"))

        fig.tight_layout()
        plt.show()

    def play_signal(self, wns):
        """
        Play the generated wind noise.

        Example: 
          wn = WindNoiseGenerator(fs=48000, duration=10)
          wn_sample = wn.generate_wind_noise()
          wn.play_signal(wn_sample)
        """

        sd.play(wns, self.fs)
        time.sleep(self.duration)
        sd.stop()

    def save_signal(self, wns, filename=None, print_log=False, num_ch=1, fs=48000):
        """
        Save the generated wind noise in a wave file.

        Example: 
          wn = WindNoiseGenerator(Fs=16000, duration=10)
          wn_sample = wn.generate_wind_noise()
          wn.save_signal(wn_sample)
        """

        if filename is None:
            local_time = time.localtime(time.time())
            filename = f"""wind_noise_{fs // 1000}kHz_
            {local_time.tm_mday}{local_time.tm_mon}
            {local_time.tm_hour}{local_time.tm_min}{local_time.tm_sec}.wav"""

        if fs != 48000:
            wns = signal.resample(wns, int(self.duration * fs))

        if num_ch == 2:
            wns = np.array([wns, wns])

        sf.write(filename, wns.T, fs)

        if print_log:
            print(f'Audio file "{filename}" saved correctly in the working directory')
