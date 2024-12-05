import numpy as np
from skimage import feature
import cv2
from skimage.feature import local_binary_pattern as lbp
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from skimage.feature import hog
from scipy.ndimage import convolve
import pywt

def a1(i, p=8, r=1):
    l = lbp(i, p, r, method="uniform")
    (h, _) = np.histogram(l.ravel(), bins=np.arange(0, p + 3), range=(0, p + 2))
    h = h.astype("float")
    h /= (h.sum() + 1e-6)
    m = np.mean(h)
    v = np.var(h)
    s = skew(h)
    k = kurtosis(h)
    return m, v, s, k

def a2(i, p=8, r=1):
    l = feature.local_binary_pattern(i, p, r, method="uniform")
    (h, _) = np.histogram(l.ravel(), bins=np.arange(0, p + 3), range=(0, p + 2))
    h = h.astype("float")
    h /= (h.sum() + 1e-6)
    u = np.sum(h**2)
    return u

def a3(i, p=8, r=1, n=3):
    l = lbp(i, p, r, method="uniform")
    (h, _) = np.histogram(l.ravel(), bins=np.arange(0, p + 3), range=(0, p + 2))
    pks, _ = find_peaks(h)
    tps = pks[np.argsort(h[pks])[-n:]][::-1]
    s = np.sort(tps)
    if len(s) < n:
        s = np.pad(s, (0, n - len(s)), 'constant', constant_values=-1)
    return s

def a4(i):
    i_rgb = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    m, v = [], []
    for j in range(3):
        ch = i_rgb[:, :, j]
        h = cv2.calcHist([ch], [0], None, [256], [0, 256])
        h = h / h.sum()
        m.append(np.mean(h))
        v.append(np.var(h))
    return m[0], v[0], m[1], v[1], m[2], v[2]

def a5(i):
    if i is None:
        print("Error: Image could not be loaded.")
        return None
    g = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    h = cv2.calcHist([g], [0], None, [256], [0, 256])
    h = h / h.sum()
    m = np.mean(h)
    v = np.var(h)
    return m, v

def a6(i):
    f, h = feature.hog(i, visualize=True)
    m = np.mean(f)
    v = np.var(f)
    return m, v

def a7(i, o=9, p=(8, 8), c=(2, 2)):
    f = hog(i, orientations=o, pixels_per_cell=p, cells_per_block=c, feature_vector=True)
    h, _ = np.histogram(f, bins=o)
    pks, _ = find_peaks(h)
    return len(pks)

def a8(i):
    sx = cv2.Sobel(i, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(i, cv2.CV_64F, 0, 1, ksize=3)
    g = np.sqrt(sx**2 + sy**2)
    return np.mean(g)

def a9(i, l=100, h=200):
    e = cv2.Canny(i, l, h)
    return np.sum(e > 0) / e.size

def a10(i, f=[0.4, 0.5, 0.6], t=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    v = []
    s = 5.0
    for fi in f:
        for th in t:
            l = 1 / fi
            k = cv2.getGaborKernel((21, 21), s, th, l, 0.5, 0, ktype=cv2.CV_64F)
            fi_img = convolve(i, k.real)
            v.append(np.var(fi_img))
    return v

def a11(i, f=[0.4, 0.5, 0.6], t=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    m = []
    s = 5.0
    for fi in f:
        for th in t:
            l = 1 / fi
            k = cv2.getGaborKernel((21, 21), s, th, l, 0.5, 0, ktype=cv2.CV_64F)
            fi_img = convolve(i, k.real)
            m.append(np.mean(fi_img))
    return m

def a12(i, pth):
    g = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    e = cv2.CascadeClassifier(pth)
    es = e.detectMultiScale(g, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    for (ex, ey, ew, eh) in es:
        er = g[ey:ey+eh, ex:ex+ew]
        _, te = cv2.threshold(er, 220, 255, cv2.THRESH_BINARY)
    return np.sum(te)

def a13(i, p=8, r=1):
    l = lbp(i, p, r, method="uniform")
    (h, _) = np.histogram(l.ravel(), bins=np.arange(0, p + 3), range=(0, p + 2))
    h = h.astype("float")
    h /= (h.sum() + 1e-6)
    return np.sum(h**2)

def a14(i):
    sx = cv2.Sobel(i, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(i, cv2.CV_64F, 0, 1, ksize=3)
    g = np.sqrt(sx**2 + sy**2)
    return np.mean(g)

def a15(i, t=100):
    s = i < t
    m = np.mean(i[s])
    v = np.var(i[s])
    return m, v

def a16(i):
    c = i.shape[1] // 2
    l = i[:, :c]
    r = i[:, c:]
    fr = np.fliplr(r)
    return np.mean((l - fr) ** 2)

def a17(i):
    coeffs = pywt.dwt2(i, 'haar')
    cA, _ = coeffs
    m = np.mean(cA)
    v = np.var(cA)
    return m, v

def a18(i):
    coeffs = pywt.dwt2(i, 'haar')
    cA, (cH, cV, cD) = coeffs
    hf = np.hstack((np.hstack((cH, cV)), cD))
    m = np.mean(hf)
    v = np.var(hf)
    return m, v

img = cv2.imread('/home/kasra/spoof_data/pre_processed_cropped_faces/train/Iman_HajMostafaee/spoof/ID_card_1.png', cv2.IMREAD_GRAYSCALE)
u = a2(img)
print("LBP Uniformity Score:", u)

d = a1(img)
print("Combined LBP Descriptor:", d)

tp = a3(img)
print("Top LBP Histogram Peaks:", tp)

img_bgr = cv2.imread('/home/kasra/spoof_data/pre_processed_cropped_faces/train/Iman_HajMostafaee/spoof/ID_card_1.png')
rm, rv, gm, gv, bm, bv = a4(img_bgr)
print("Red Mean:", rm, "Red Variance:", rv)
print("Green Mean:", gm, "Green Variance:", gv)
print("Blue Mean:", bm, "Blue Variance:", bv)

m, v = a5(img_bgr)
print("Grayscale Histogram Mean:", m, "Grayscale Histogram Variance:", v)

hm, hv = a6(img)
print("HOG Mean:", hm, "HOG Variance:", hv)

pk = a7(img)
print("Number of Significant HOG Histogram Peaks:", pk)

ed_s = a8(img)
print("Edge Density (Sobel Filter):", ed_s)

ed_c = a9(img)
print("Edge Density (Canny Filter):", ed_c)

gr = a10(img)
print("Gabor Filter Response Variances:", gr)

gm = a11(img)
print("Gabor Filter Mean Values:", gm)

ld = a12(img, '/home/kasra/spoof_data/cascades/haarcascade_frontalface_default.xml')
print("Detected Faces Thresholding Sum:", ld)

lbp_sum = a13(img)
print("LBP Histogram Sum:", lbp_sum)

grad = a14(img)
print("Gradient Magnitude Mean:", grad)

tm, tv = a15(img)
print("Thresholded Image Mean:", tm, "Thresholded Image Variance:", tv)

df = a16(img)
print("Image Difference (Left-Right):", df)

wtm, wtv = a17(img)
print("Wavelet Transform Mean:", wtm, "Wavelet Transform Variance:", wtv)

wtm2, wtv2 = a18(img)
print("Wavelet Transform (Detailed) Mean:", wtm2, "Wavelet Transform (Detailed) Variance:", wtv2)
