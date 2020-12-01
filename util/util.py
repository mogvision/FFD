import cv2
import numpy as np
import sys
import time

def readkp(filename):
    def is_float(n):
        try:
            float(n)
            return True
        except:
            return False
    kp_temp = []
    counter = 0
    num_kp = 0
    with open(filename, "r") as file:
        for line in file.readlines():
            counter += 1

            if (counter == 2):
                num_kp = int(line)
            if (counter > 2):
                kp_temp.append( [float(n) for n in line.split(',') if is_float(n)] )

    kp_temp = np.array(kp_temp, dtype=np.float32)

    assert (kp_temp.shape[0] == num_kp)
    assert (kp_temp.shape[1] == 4)

    return kp_temp, num_kp


def KP_opencv(np_kp_matrix):
    np_kp_matrix = np.array(np_kp_matrix, dtype=np.float32)
    keypoints_opencv = []
    for i in range(np_kp_matrix.shape[0]):
        keypoints_opencv.append(cv2.KeyPoint(x=np_kp_matrix[i,0],
            y=np_kp_matrix[i,1],
            _size=np_kp_matrix[i,2],
            _response=np_kp_matrix[i,3],
            _angle=-1,
            _octave=0,
            _class_id=0))
    return keypoints_opencv 

def RootSIFT_des(gray, kpts):
    kpts_cv = KP_opencv(kpts)
    sift = cv2.xfeatures2d.SIFT_create()

    kp, des_sift= sift.compute(gray, kpts_cv)     
    des_sift /= (des_sift.sum(axis=1, keepdims=True) + sys.float_info.epsilon)
    des = np.sqrt(des_sift)
    return des, kp




def drawKeyPts(im, keyp, color, th, im_save):
    for kp_i in keyp:
        center = (int(np.round(kp_i.pt[0])), int(np.round(kp_i.pt[1])))
        radius = int(np.round(kp_i.size*2.))
        cv2.circle(im, center, radius, color, thickness=th)
        
        orient = (int(np.round(np.cos(kp_i.angle)*radius)), int(np.round(np.sin(kp_i.angle)*radius)))
        cv2.line(im, center, (center[0]+orient[0], center[1]+orient[1]), color, 1)
    
    cv2.imwrite(im_save, im) 



def Matching_FFD_keypoints(IMGs, KPTS_FFD, num_show_matches):
    img1 = cv2.imread(IMGs[0])  
    img2 = cv2.imread(IMGs[1]) 

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    des1, kp1 = RootSIFT_des(gray1, KPTS_FFD[0])
    des2, kp2 = RootSIFT_des(gray2, KPTS_FFD[1])

    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)


    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[0:num_show_matches], img2, flags=2)
    kp_num = np.minimum(KPTS_FFD[0].shape[0], KPTS_FFD[1].shape[0])
    print("[+] #Detected keypoints:  %d -> #matches: %d  (%0.1f%%)"%(kp_num, len(matches), 100.*len(matches)/kp_num))

    drawKeyPts(img1, kp1, (0,255,0), 2, "kp1.png")
    drawKeyPts(img2, kp2, (0,0,255), 2, "kp2.png")
    cv2.imwrite("matched_images.png", img3) 
