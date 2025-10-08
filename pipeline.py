import cv2
from time import perf_counter
from utils import read_gray

def match_orb_score(img1_path, img2_path, ratio=0.75, nfeatures=1000):
    img1 = read_gray(img1_path)
    img2 = read_gray(img2_path)

    t0 = perf_counter()
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0, (perf_counter() - t0) * 1000.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    t1 = perf_counter()
    return len(good), (t1 - t0) * 1000.0

def match_sift_score(img1_path, img2_path, ratio=0.75):
    img1 = read_gray(img1_path)
    img2 = read_gray(img2_path)

    t0 = perf_counter()
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0, (perf_counter() - t0) * 1000.0

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    t1 = perf_counter()
    return len(good), (t1 - t0) * 1000.0