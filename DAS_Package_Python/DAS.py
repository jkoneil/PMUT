import numpy as np

def DAS(rf, no_ele, Fs, channel_spacing, speed_sound):

    ns, nl = rf.shape # samples, elements

    sample_spacing = (1 / Fs * speed_sound) * 1000 # mm

    postBF = np.zeros((ns, nl))

    for n in range(nl): # lateral pixel index
        for m in range(ns): # axial pixel index
            for i in range(no_ele): #sensor index
                if 0 <= i < nl:
                    depth = (m+1) *sample_spacing
                    width = abs(i - n) * channel_spacing
                    distance = np.sqrt(depth**2 + width**2)

                    delay = distance / sample_spacing

                    if delay < ns:
                        idx = int(round(delay))
                        if idx < ns:
                            y = rf[idx, i]
                            if np.isnan(y):
                                y = 0
                            postBF[m, n] += y
    return postBF