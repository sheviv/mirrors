import argparse
import cv2
import pyaudio
import wave

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--s', default=5, help='record seconds')
args = vars(parser.parse_args())
args["s"] = int(args["s"])

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = args["s"] + 1
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("*recording*")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
# while True:
# for i in range(0, int(RATE / CHUNK * int("inf"))):
#     data = stream.read(CHUNK)
#     frames.append(data)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    # if str(input()) == str('q'):
        # stream.stop_stream()
        # stream.close()
        # p.terminate()
        # break

print("*done recording*")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
