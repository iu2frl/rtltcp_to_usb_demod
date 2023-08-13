import socket
import logging
import numpy as np
import scipy.signal
import datetime
import struct
import json
import scipy.signal
import scipy.io.wavfile
import threading

### PARAMETERS TO BE SET ###
RTLTCP_ADDRESS = "192.168.0.103" # Address of rtl_tcp server
RTLTCP_PORT = 1234 # Port of rtl_tcp server
RTLTCP_SAMPLERATE = 2048000 # Samplerate in samples/s
RTLTCP_TUNINGFREQ = 144500000 # Center frequency of the SDR receiver
RTLTCP_USBSIG = 144174000 # Carrier frequency of the USB signal
RTLTCP_GAIN = -1 # -1: AGC, positive numbers: desired gain
### END ###

# RTL_TCP commands
# https://github.com/osmocom/rtl-sdr/blob/master/src/rtl_tcp.c#L502
SET_FREQUENCY = 0x01
SET_SAMPLERATE = 0x02
SET_GAINMODE = 0x03
SET_GAIN = 0x04
SET_FREQENCYCORRECTION = 0x05
SET_AGCMODE = 0x08
# Set logging output
logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger('findfont').setLevel(logging.INFO)
# Remote socket to connect
serverSocket = None
# Connect to the IQ server
# https://github.com/pyrtlsdr/pyrtlsdr
class RtlSdrTcpClient:
    # Import default values
    global RTLTCP_ADDRESS, RTLTCP_PORT
    sdrSocket: socket = None
    sdrHostname: str = ""
    sdrPort: int = 0
    # Init RTL_TCP class
    def __init__(self, hostname=RTLTCP_ADDRESS, port=RTLTCP_PORT):
        self.sdrHostname = hostname
        self.sdrPort = port
        self.sdrSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sdrSocket.connect((self.sdrHostname, self.sdrPort))
        self._tune()
    # Read samples command
    def read_samples(self):
        msg = json.dumps({'type': 'method', 'name': 'read_samples'})
        header = struct.pack('!I', len(msg))
        self.sdrSocket.send(header + msg.encode())
        data_len = struct.unpack('!I', self.sdrSocket.recv(4))[0]
        data = self.sdrSocket.recv(data_len)
        return data
    # Close TCP connection
    def close(self):
        self.sdrSocket.close()
        self.sdrSocket = None
    # Send SDR parameters to server
    def _tune(self):
        try:
            self.sdrSocket.send(struct.pack(">BI", SET_FREQUENCY, RTLTCP_TUNINGFREQ))
            self.sdrSocket.send(struct.pack(">BI", SET_SAMPLERATE, RTLTCP_SAMPLERATE))
            if RTLTCP_GAIN > 0:
                self.sdrSocket.send(struct.pack(">BI", SET_GAIN, RTLTCP_GAIN))
                self.sdrSocket.send(struct.pack(">BI", SET_GAINMODE, 1))
            else:
                self.sdrSocket.send(struct.pack(">BI", SET_GAINMODE, 0))
        except:
            self.sdrSocket = None
# Draw spectrogram of IQ signal
def DrawSpectrogram(iqSignal, samplingRate):
    import matplotlib.pyplot as plt
    # Create a spectrogram
    Pxx, freqs, times, im = plt.specgram(iqSignal, Fs=samplingRate, NFFT=1024)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.colorbar(label='Intensity (dB)')
    plt.show()
# Draq complex FFT of IQ signal
def DrawPSD(iqSignal, samplingRate):
    import matplotlib.pyplot as plt
    # Calculate the PSD
    f, Pxx = scipy.signal.welch(iqSignal, fs=samplingRate, nperseg=1024)
    # Visualize the PSD
    plt.semilogy(f, Pxx)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency')
    plt.title('Power Spectral Density')
    plt.show()
# Draw real FFT of IQ signal
def DrawPSD2(iqSignal, samplingRate):
    import numpy as np
    import matplotlib.pyplot as plt

    # Assume iq_signal is your complex IQ signal
    # Sample rate and FFT parameters
    sampling_rate = samplingRate  # Your sampling rate
    fft_size = len(iqSignal)  # Size of the FFT

    # Compute the FFT of the IQ signal
    fft_result = np.fft.fft(iqSignal, fft_size)
    frequencies = np.fft.fftfreq(fft_size, d=1/sampling_rate)

    # Plot the positive half of the spectrum
    positive_frequencies = frequencies[:fft_size // 2]
    positive_spectrum = np.abs(fft_result[:fft_size // 2])

    plt.figure(figsize=(10, 6))
    plt.plot(positive_frequencies, positive_spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Positive Frequency Spectrum')
    plt.grid()
    plt.show()
# Demodulate USB signal from IQ
def DecodeUsbSignal(inputIq, output_file: str, carrierFreq: int):
    logging.debug("DEC -> Starting USB decode")
    # Create the carrier signal for demodulation
    sampling_rate = RTLTCP_SAMPLERATE  # Adjust this to your actual sampling rate
    t = np.arange(0, len(inputIq)) / sampling_rate
    #carrier_freq = 0  # Adjust this to your USB carrier frequency
    carrier_signal = np.exp(1j * 2 * np.pi * carrierFreq * t)
    # Demodulate the USB signal
    demodulated_signal = inputIq * carrier_signal
    # # Shift the spectrum down
    # logging.debug(f"DEC -> Shifting IQ down {carrierFreq} Hz")
    # demodulated_signal = ShiftIQ(demodulated_signal,    carrierFreq)
    #Ensure that the demodulated signal is real-valued
    demodulated_signal = demodulated_signal.real
    # Normalize the signal before scaling
    max_value = np.max(np.abs(demodulated_signal))
    if max_value != 0:
        normalized_signal = demodulated_signal / max_value
    else:
        normalized_signal = demodulated_signal
    # Scale the signal for 16-bit audio
    scaled_signal = np.int16(normalized_signal * 32767)
    # Write to file
    logging.debug(f"DEC -> Writing to {output_file}")
    scipy.io.wavfile.write(output_file, sampling_rate, scaled_signal)
    logging.debug(f"DEC -> Writing completed")
# Convert bytearray to ndarray
def PackedBytesToIQ(bytes_data):
    """Adapted version of packed_bytes_to_iq to handle provided data
    """
    data = np.frombuffer(bytes_data, dtype=np.int8)  # Convert bytes to int8 array
    iq_real = data[::2]  # Real part of IQ samples (even indices)
    iq_imag = data[1::2]  # Imaginary part of IQ samples (odd indices)
    # Make sure arrays have same size
    if len(iq_real) > len(iq_imag):
        iq_real = iq_real[:len(iq_imag)]
    elif len(iq_imag) > len(iq_real):
        iq_imag = iq_imag[:len(iq_real)]
    # Combine real and imaginary parts to form complex numbers
    iq = iq_real + 1j * iq_imag      
    iq /= 127.5
    iq -= (1 + 1j)
    # Return values
    return iq
# Apply low pass filter to IQ (typical to 3000Hz of SSB)
def IQLowPassFilter(iq_signal, minFreq, maxFreq, sampling_rate):
    """Apply a bandpass filter to an IQ signal to extract frequencies between minFreq and maxFreq.
    """
    nyquist_rate = 0.5 * sampling_rate
    min_norm_freq = minFreq / nyquist_rate
    max_norm_freq = maxFreq / nyquist_rate
    #b, a = scipy.signal.butter(6, [min_norm_freq, max_norm_freq], btype='band')
    b, a = scipy.signal.butter(6, max_norm_freq, btype='low')
    filtered_signal = scipy.signal.lfilter(b, a, iq_signal)
    return filtered_signal
# Filter IQ signal and send it to USB decoder
def FilterAndDecode(inputIq: np.ndarray[any], fileName: str):
    # Assuming you have your IQ signal stored in the variable "iq_signal"
    samplingRate = RTLTCP_SAMPLERATE  # Sampling rate in Hz
    carrierFreq = (samplingRate/4) - (RTLTCP_TUNINGFREQ - RTLTCP_USBSIG) 
    minFreq = int(carrierFreq)  # Minimum frequency in Hz
    maxFreq = int(carrierFreq + 3000)  # Maximum frequency in Hz
    #logging.debug("DEC -> Plotting inputIq spectrum")
    #DrawPSD2(inputIq, samplingRate)
    #DrawSpectrogram(inputIq, samplingRate)
    logging.debug(f"DEC -> Tuning freq: {RTLTCP_TUNINGFREQ} Hz, FT8: {RTLTCP_USBSIG} Hz")
    logging.debug(f"DEC -> USB IQ carrier frequency at {carrierFreq} Hz")
    logging.debug(f"DEC -> Applying IQ filter from {minFreq} Hz to {maxFreq} Hz")
    filteredSignal = IQLowPassFilter(inputIq, minFreq, maxFreq, samplingRate)
    #logging.debug("DEC -> Plotting filtered IQ bandwidth")
    #DrawPSD2(filtered_signal, samplingRate)
    #DrawSpectrogram(filtered_signal, samplingRate)
    # Now "filtered_signal" contains the IQ signal with frequencies between minFreq and maxFreq
    DecodeUsbSignal(filteredSignal, fileName, minFreq)
# Connect to the IQ server and process samples
def Main():
    logging.debug("IQ -> Initializing IQ reception")
    # Create sockets
    global serverSocket
    startSampling = False
    # Initialize SDR object
    sdrClient = RtlSdrTcpClient()
    iqBuffer = np.array([], dtype=np.complex128)  # Initialize an empty complex array
    fileName: str = ""
    while True:
        if sdrClient.sdrSocket is None:
            # Create an instance of the RtlSdrTcpClient class
            sdrClient = RtlSdrTcpClient()
        else:
            # Read samples from the RTL-SDR device
            tcpData = sdrClient.read_samples()
            # Start USB decoding every 15 seconds (FT8 timing)
            if datetime.datetime.now().second % 15 == 0 and not startSampling:
                # Start FT8 reception
                decodeStart = datetime.datetime.now()
                startSampling = True
                iqBuffer = np.array([], dtype=np.complex128)  # Clear the buffer for new reception
                fileName = datetime.datetime.now().strftime("%H%m%Y_%H%M%S.wav")
                logging.debug("IQ -> Starting IQ reception")
            if startSampling:
                # End of receive time windows
                if (datetime.datetime.now() - decodeStart).total_seconds() >= 14:
                    startSampling = False
                    logging.debug("IQ -> Completed IQ reception")
                    if len(iqBuffer) > 10:
                        decodeThread = threading.Thread(target=FilterAndDecode, args=(iqBuffer, fileName, ), name=f"Decode_{fileName}")
                        decodeThread.start()
                else:
                    # Store received samples
                    if len(tcpData) > 0:
                        # Interpret binary data as IQ samples and append to the buffer
                        iqPackets = PackedBytesToIQ(tcpData)
                        iqBuffer = np.append(iqBuffer, iqPackets)

# Execute main routine
Main()
