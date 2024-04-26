import cv2
import time
import av
import zlib

def bench_compress():
    # Create a video stream
    capture = cv2.VideoCapture("./vid.mp4")
    container = av.open('output_stream.hevc', mode='w')
    stream = container.streams.video[0]

    # Read all frames
    frames = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            frames.append(frame)

    # Compress all frames
    zlib_times = []
    zlib_compressed_frames = []
    for frame in frames:
        start = time.time()
        compressed_frame = zlib.compress(frame.to_rgb().to_bytes())
        end = time.time()
        zlib_times.append(end - start)
        zlib_compressed_frames.append(compressed_frame)

    for frame in frames:
        start = time.time()
        compressed_frame = zlib.compress(frame.to_rgb().to_bytes())
        end = time.time()
        zlib_times.append(end - start)
        zlib_compressed_frames.append(compressed_frame)
