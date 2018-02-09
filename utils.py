import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import matplotlib.lines as mlines
from matplotlib import transforms
import argparse, os, fnmatch, shutil
import numpy as np
import cv2
import math
import copy
import librosa
import dlib
import subprocess
from keras import backend as K
from tqdm import tqdm

font = {'size'   : 18}
mpl.rc('font', **font)

# Lookup tables for drawing lines between points
Mouth = [[48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], \
         [57, 58], [58, 59], [59, 48], [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], \
         [66, 67], [67, 60]]

Nose = [[27, 28], [28, 29], [29, 30], [30, 31], [30, 35], [31, 32], [32, 33], \
        [33, 34], [34, 35], [27, 31], [27, 35]]

leftBrow = [[17, 18], [18, 19], [19, 20], [20, 21]]
rightBrow = [[22, 23], [23, 24], [24, 25], [25, 26]]

leftEye = [[36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [36, 41]]
rightEye = [[42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [42, 47]]

other = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], \
         [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], \
         [12, 13], [13, 14], [14, 15], [15, 16]]

faceLmarkLookup = Mouth + Nose + leftBrow + rightBrow + leftEye + rightEye + other

class faceNormalizer(object):
    # Credits: http://www.learnopencv.com/face-morph-using-opencv-cpp-python/
    w = 600
    h = 600

    def __init__(self, w = 600, h = 600):
        self.w = w
        self.h = h

    def similarityTransform(self, inPoints, outPoints):
        s60 = math.sin(60*math.pi/180)
        c60 = math.cos(60*math.pi/180)
      
        inPts = np.copy(inPoints).tolist()
        outPts = np.copy(outPoints).tolist()
        
        xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
        yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
        
        inPts.append([np.int(xin), np.int(yin)])
        
        xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
        yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
        
        outPts.append([np.int(xout), np.int(yout)])
        
        tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)
        
        return tform

    def tformFlmarks(self, flmark, tform):
        transformed = np.reshape(np.array(flmark), (68, 1, 2))           
        transformed = cv2.transform(transformed, tform)
        transformed = np.float32(np.reshape(transformed, (68, 2)))
        return transformed

    def alignEyePoints(self, lmarkSeq):
        w = self.w
        h = self.h

        alignedSeq = copy.deepcopy(lmarkSeq)
        firstFlmark = alignedSeq[0,:,:]
        
        eyecornerDst = [ (np.float(0.3 * w ), np.float(h / 3)), (np.float(0.7 * w ), np.float(h / 3)) ]
        eyecornerSrc  = [ (firstFlmark[36, 0], firstFlmark[36, 1]), (firstFlmark[45, 0], firstFlmark[45, 1]) ]

        tform = self.similarityTransform(eyecornerSrc, eyecornerDst);

        for i, lmark in enumerate(alignedSeq):
            alignedSeq[i] = self.tformFlmarks(lmark, tform)

        return alignedSeq

    def alignEyePointsV2(self, lmarkSeq):
        w = self.w
        h = self.h

        alignedSeq = copy.deepcopy(lmarkSeq)
        
        eyecornerDst = [ (np.float(0.3 * w ), np.float(h / 3)), (np.float(0.7 * w ), np.float(h / 3)) ]
    
        for i, lmark in enumerate(alignedSeq):
            curLmark = alignedSeq[i,:,:]
            eyecornerSrc  = [ (curLmark[36, 0], curLmark[36, 1]), (curLmark[45, 0], curLmark[45, 1]) ]
            tform = self.similarityTransform(eyecornerSrc, eyecornerDst);
            alignedSeq[i,:,:] = self.tformFlmarks(lmark, tform)

        return alignedSeq

    def unitNorm(self, flmarkSeq):
        normSeq = copy.deepcopy(flmarkSeq)
        normSeq[:, : , 0] /= self.w
        normSeq[:, : , 1] /= self.h
        return normSeq

def write_video_wpts_wsound_unnorm(frames, sound, fs, path, fname, xLim, yLim):
    try:
        os.remove(os.path.join(path, fname+'.mp4'))
        os.remove(os.path.join(path, fname+'.wav'))
        os.remove(os.path.join(path, fname+'_ws.mp4'))
    except:
        print 'Exp'

    if len(frames.shape) < 3:
        frames = np.reshape(frames, (frames.shape[0], frames.shape[1]/2, 2))
    print frames.shape

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=25, metadata=metadata)

    fig = plt.figure(figsize=(10, 10))
    l, = plt.plot([], [], 'ko', ms=4)


    plt.xlim(xLim)
    plt.ylim(yLim)

    librosa.output.write_wav(os.path.join(path, fname+'.wav'), sound, fs)

    lines = [plt.plot([], [], 'k')[0] for _ in range(3*len(dt))]

    with writer.saving(fig, os.path.join(path, fname+'.mp4'), 150):
        plt.gca().invert_yaxis()
        for i in tqdm(range(frames.shape[0])):
            l.set_data(frames[i,:,0], frames[i,:,1])
            cnt = 0
            for refpts in faceLmarkLookup:
                lines[cnt].set_data([frames[i,refpts[1], 0], frames[i,refpts[0], 0]], [frames[i, refpts[1], 1], frames[i,refpts[0], 1]])
                cnt+=1
            writer.grab_frame()

    cmd = 'ffmpeg -i '+os.path.join(path, fname)+'.mp4 -i '+os.path.join(path, fname)+'.wav -c:v copy -c:a aac -strict experimental '+os.path.join(path, fname)+'_ws.mp4'
    subprocess.call(cmd, shell=True) 
    print('Muxing Done')

    os.remove(os.path.join(path, fname+'.mp4'))
    os.remove(os.path.join(path, fname+'.wav'))

def write_video_wpts_wsound(frames, sound, fs, path, fname, xLim, yLim):
    try:
        os.remove(os.path.join(path, fname+'.mp4'))
        os.remove(os.path.join(path, fname+'.wav'))
        os.remove(os.path.join(path, fname+'_ws.mp4'))
    except:
        print 'Exp'

    if len(frames.shape) < 3:
        frames = np.reshape(frames, (frames.shape[0], frames.shape[1]/2, 2))
    print frames.shape

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=25, metadata=metadata)

    fig = plt.figure(figsize=(10, 10))
    l, = plt.plot([], [], 'ko', ms=4)


    plt.xlim(xLim)
    plt.ylim(yLim)

    librosa.output.write_wav(os.path.join(path, fname+'.wav'), sound, fs)

    rect = (0, 0, 600, 600)
    
    if frames.shape[1] == 20:
        lookup = [[x[0] - 48, x[1] - 48] for x in Mouth]
        print lookup
    else:
        lookup = faceLmarkLookup

    lines = [plt.plot([], [], 'k')[0] for _ in range(3*len(lookup))]

    with writer.saving(fig, os.path.join(path, fname+'.mp4'), 150):
        plt.gca().invert_yaxis()
        for i in tqdm(range(frames.shape[0])):
            l.set_data(frames[i,:,0], frames[i,:,1])
            cnt = 0
            for refpts in lookup:
                lines[cnt].set_data([frames[i,refpts[1], 0], frames[i,refpts[0], 0]], [frames[i, refpts[1], 1], frames[i,refpts[0], 1]])
                cnt+=1
            writer.grab_frame()

    cmd = 'ffmpeg -y -i '+os.path.join(path, fname)+'.mp4 -i '+os.path.join(path, fname)+'.wav -c:v copy -c:a aac -strict experimental '+os.path.join(path, fname)+'_ws.mp4'
    subprocess.call(cmd, shell=True) 
    print('Muxing Done')

    os.remove(os.path.join(path, fname+'.mp4'))
    os.remove(os.path.join(path, fname+'.wav'))


def plot_flmarks(pts, lab, xLim, yLim, xLab, yLab, figsize=(10, 10)):
    if len(pts.shape) != 3:
        pts = np.reshape(pts, (pts.shape[0]/2, 2))
    print pts.shape

    if pts.shape[0] == 20:
        lookup = [[x[0] - 48, x[1] - 48] for x in Mouth]
        print lookup
    else:
        lookup = faceLmarkLookup

    plt.figure(figsize=figsize)
    plt.plot(pts[:,0], pts[:,1], 'ko', ms=4)
    for refpts in lookup:
        plt.plot([pts[refpts[1], 0], pts[refpts[0], 0]], [pts[refpts[1], 1], pts[refpts[0], 1]], 'k', ms=4)

    plt.xlabel(xLab, fontsize = font['size'] + 4, fontweight='bold')
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top') 
    plt.ylabel(yLab, fontsize = font['size'] + 4, fontweight='bold')
    plt.xlim(xLim)
    plt.ylim(yLim)
    plt.gca().invert_yaxis()
   
    plt.savefig(lab, dpi = 300, bbox_inches='tight')
    plt.clf()
    plt.close()

def melSpectra(y, sr, wsize, hsize):
    cnst = 1+(int(sr*wsize)/2)
    y_stft_abs = np.abs(librosa.stft(y,
                                  win_length = int(sr*wsize),
                                  hop_length = int(sr*hsize),
                                  n_fft=int(sr*wsize)))/cnst

    melspec = np.log(librosa.feature.melspectrogram(sr=sr, 
                                             S=y_stft_abs**2,
                                             n_mels=64))
    return melspec

def main():
    return

if __name__ == "__main__":
    main()