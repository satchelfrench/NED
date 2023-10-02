
import subprocess
import importlib.util
import numpy as np
import cv2

def _check_and_install(name: str) -> bool:
    spec = importlib.util.find_spec(name)
    if spec is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", name])
        spec = importlib.util.find_spec(name)

    return (spec is not None)


def calc_optical_flow_dense(prior, current):
    frame1 = cv2.cvtColor(prior, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def flow_to_rgb(flow):
    hsv = np.zeros_like(prior)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[...,0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return rgb

