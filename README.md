# rtl_tcp to USB WAV file

Demodulates a TCP IQ stream from rtl_tcp to a WAV file of the USB signal

## Usage

1. Set receiver parameters in `iqstream_to_usb.py`
2. (optional) Create virtual environment `python3 -m venv venv`
3. (optional) Activate virtual envirotnment `source venv/bin/activate`
4. Install requirements `pip install -r requirements.txt`
5. Make sure __rtl_tcp__ is running on a reachable IP address
6. Execute the main code `python3 ./iqstream_to_usb.py`

## Description

- The script opens a socket to the rtl_tcp server
- Tuning parameters are sent to the server (gain, tuning freq and samplerate)
- TCP packets are decoded and converted to `numpy.ndarray`
- `numpy.ndarray` is filtered to `3000 Hz` (maximum USB signal width for ham radio bands)
- filtered signal is processed to extract the USB signal and written to WAV file

## Known bugs 

Help is appreciated here!

1. Output WAV file has a samplerate of `RTLTCP_SAMPLERATE`, this is way too high
2. Output WAV file is not shifted to zero, this means:
    - Assume `RTLTCP_TUNINGFREQ` is `144500000` and `RTLTCP_USBSIG` is `144174000`, with `RTLTCP_SAMPLERATE` set to `2048000`
    - This means that we have a usable bandwidth of `2048000 / 2` which is `1024000 Hz` around the `RTLTCP_TUNINGFREQ`, meaning we can listen to any signal from `143988000 Hz` to `145012000 Hz`
    - If we do some math and calculate the USB carrier frequency as `(RTLTCP_SAMPLERATE / 4) - (RTLTCP_TUNINGFREQ - RTLTCP_USBSIG)` we have the the carrier at `186000 Hz` over the zero frequency
    - If we open the WAV file with Audacity and check the audio spectrum we notice that the demodulated signal is exactly at `186000 Hz`, meaning that the signal needs to be shifted down by a "carrier frequency" amount. Suggestions are welcome!
