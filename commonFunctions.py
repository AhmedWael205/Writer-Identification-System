import os
import random
from shutil import copyfile,rmtree
import skimage.io as io
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.pyplot import bar
from scipy.signal import find_peaks
from sklearn.svm import SVC
import glob
from skimage.morphology import convex_hull_image

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import time

######################################################################################################

def set_key(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = [value]
    elif type(dictionary[key]) == list:
        dictionary[key].append(value)
    else:
        dictionary[key] = [dictionary[key], value]

def SortDataset(formFile,datasetPath,outPath):

    file1 = open(formFile, 'r') 
    Lines = file1.readlines()

    d = {}

    for Line in Lines:
            x = Line.split()
            if "#" not in x[0]:
                set_key(d, x[1], x[0])

    if os.path.exists(outPath) and os.path.isdir(outPath):
        rmtree(outPath)

    try:  
        os.mkdir(outPath)
    except OSError as error:  
        print("Failed")
        return

    for key, values in d.items():
        path = outPath+key+'/'
        try:  
            os.mkdir(path)
        except OSError as error:  
            pass
        values = np.asarray(values)
        if len(values) > 0:
            for value in values:
                try:
                    copyfile(datasetPath+value+".png", path+value+".png")
                except:
                    print("Failed to copy")
                    pass
    print("Done")
        

def GenerateNSamples(N,sortedDataset,outSamplePath):
    y = []

    x = sorted(glob.glob(sortedDataset+"*"))

    if os.path.exists(outSamplePath) and os.path.isdir(outSamplePath):
        rmtree(outSamplePath)

    try:  
        os.mkdir(outSamplePath)
    except OSError as error:  
        pass

    for n in range(N):
        writer = [0,0,0]
        forms = [[],[],[]]

        writer[0] = random.randint(0,len(x)-1)
        forms[0] = sorted(glob.glob(x[writer[0]]+"/*"))
        while len(forms[0]) < 3:
            writer[0] = random.randint(0,len(x)-1)
            forms[0] = sorted(glob.glob(x[writer[0]]+"/*"))

        writer[1] = random.randint(0,len(x)-1)
        forms[1] = sorted(glob.glob(x[writer[1]]+"/*"))
        while len(forms[1]) < 3 or writer[1] == writer[0]:
            writer[1] = random.randint(0,len(x)-1)
            forms[1] = sorted(glob.glob(x[writer[1]]+"/*"))

        writer[2] = random.randint(0,len(x)-1)
        forms[2] = sorted(glob.glob(x[writer[2]]+"/*"))
        while len(forms[2]) < 3 or writer[1] == writer[2] or writer[0] == writer[2]:
            writer[2] = random.randint(0,len(x)-1)
            forms[2] = sorted(glob.glob(x[writer[2]]+"/*"))


        y.append(random.randint(0,2))

        if n < 9:
            k = "0" + str(n+1)
        else:
            k = str(n+1)

        print(k)
        path = outSamplePath+"/"+k+"/"

        try:  
            os.mkdir(path)
        except OSError as error:  
            pass

        for i in range(len(writer)):

            try:  
                os.mkdir(path+str(i+1))
            except OSError as error:  
                pass

            t1 = random.randint(0,len(forms[i])-1)
            t2 = random.randint(0,len(forms[i])-1)

            while t2 == t1:
                t2 = random.randint(0,len(forms[i])-1)

            copyfile(forms[i][t1], path+str(i+1)+"/1.png")
            copyfile(forms[i][t2], path+str(i+1)+"/2.png")

            if i == y[n]:
                t3 = random.randint(0,len(forms[i])-1)
                while t3 == t1 or t3 == t2:
                    t3 = random.randint(0,len(forms[i])-1)
                copyfile(forms[i][t3], path+"test.png")

    f = open(outSamplePath+"/out.txt", "a")
    for i in range(len(y)):
        f.write(str(y[i]+1)+"\n")
    f.close()

######################################################################################################


def FeatureExtraction(BS, Bin,All = False):
    ImageFeatureVector = []

    f9 = InkDensity(Bin) #Global
    for s in BS:

        TL, BL, UL, LL = getLines(s)

        #Features
        f1 = abs(TL - UL)
        f2 = abs(UL - LL)
        f3 = abs(LL - BL)
        if f2 == 0:
            f2 = 0.001
        if f3 == 0:
            f3 = 0.001

        f4 = f1/f2
        f5 = f1/f3
        f6 = f2/f3

        f7 = TransitionMedian(s)
        if f7 == 0:
            f7 = 0.001
        f8 = f2/f7


        f10 = AvgSpaces(s)
        ImageFeatureVector.append([f4,f5,f6,f8,f9,f10])

        if All:
            f11 = SlantAnglePDF(s)
            ImageFeatureVector.append([f4,f5,f6,f8,f9,f10,f11])
        
    return ImageFeatureVector

def SlantAngle(mask):
    hist = np.zeros((1, 9))
    if mask[2, 2] == 0:
        return hist
    hist[0][0] = mask[3, 2] & mask[4, 2]
    hist[0][1] = mask[3, 1] & mask[4, 1]
    hist[0][2] = mask[3, 1] & mask[4, 0]
    hist[0][3] = mask[2, 1] & mask[3, 0]
    hist[0][4] = mask[2, 1] & mask[2, 0]
    hist[0][5] = mask[2, 1] & mask[1, 0]
    hist[0][6] = mask[2, 1] & mask[0, 0]
    hist[0][7] = mask[1, 1] & mask[0, 1]
    hist[0][8] = mask[1, 2] & mask[0, 2]
    return hist

def SlantAnglePDF(s):
    PDF = np.zeros((1, 9))
    x, y = s.shape
    s = s//255
    for i in range(2, x - 2):
        for j in range(2, y - 2):
            mask = s[i - 2:i + 3, j - 2:j + 3]
            m= SlantAngle(mask)
            PDF = PDF + m
    return PDF

def getLines(sentence):
    HorizontalProjMax = np.max(sentence, axis=1)
    TopLine = np.argwhere(HorizontalProjMax == 255)[0][0]
    BottomLine = np.argwhere(HorizontalProjMax == 255)[-1][0]

    HorizontalProjSum = np.sum(sentence, axis=1)
    peak = np.max(HorizontalProjSum)
    Index = np.argmax(HorizontalProjSum)

    UpperLine = Index - 1
    LowerLine = Index + 1

    while HorizontalProjSum[UpperLine] > peak/2 and UpperLine >= 0:
        UpperLine -= 1
    while HorizontalProjSum[LowerLine] > peak/2 and LowerLine < sentence.shape[0]-1:
        LowerLine += 1

    return TopLine, BottomLine, UpperLine, LowerLine

def InkDensity(Bin):
    return (np.sum(Bin)/(Bin.shape[0]*Bin.shape[1]))

def TransitionMedian(s):
    HorizontalProjSum = np.sum(s, axis=1)
    Index = np.argmax(HorizontalProjSum)
    row = s[Index,:]
    BlackRuns = []
    c = 0
    for col in row:
        if (col == 0):
            c += 1
        else:
            if (c != 0):
                BlackRuns.append(c)
            c = 0
    BlackRuns = np.asarray(BlackRuns)
    median = np.median(BlackRuns)
    return median

def AvgSpaces(s):
    VerticalProjection = np.sum(s, axis=0)
    BlackCol = np.where(VerticalProjection == 0)
    ColumnDiff = np.diff(BlackCol)
    ColumnDiff[ColumnDiff < 30] = 0
    return np.average(ColumnDiff)

################################################################################################################

def calcAccuarcy(Predict='out.txt', Actual='TestSet/out.txt'):
    file1 = open(Predict, 'r')
    file2 = open(Actual, 'r')

    Lines1 = file1.readlines()
    Lines2 = file2.readlines()

    file1.close()
    file2.close()

    wrong = 0
    for i in range(len(Lines1)):
        try:
            if Lines1[i][0] != Lines2[i][0]:
                print(i+1)
                wrong += 1
        except:
            pass
            
    acc = 1 - wrong/len(Lines1)
    print("Accuarcy = "+str(acc*100)+"%")


def show_images(images,titles=None,saveImage=False,name="untitled",axis=False):
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    if not axis:
        plt.axis("off")
    else:
        plt.axis("on")
    plt.show()
    if saveImage:
        show_images.counter +=1
        path = "temp/"
        x = path + name + "_" + str(show_images.counter) + ".png"
        fig.savefig(x, bbox_inches='tight',dpi=240)
show_images.counter = 0

def Preprocessing(img,num=0):
    if num == 2:
        return AccuratePreprocessing(img)
    elif num ==1:
        return SemiPreprocessing(img)

    return FastPreprocessing(img)


def AccuratePreprocessing(img):
    x, y = img.shape

    blur = cv.GaussianBlur(img,(11,11),0)
    _,Thresh = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    negThresh = np.max(Thresh) - Thresh

    StartVertical = np.where(negThresh[0, :] == 0)[0][0] * 5
    lines = cv.HoughLinesP(negThresh, 1, math.pi/2, 3, 3, 200, 1)

    i = 0
    lines_y = []
    for line in lines:
        if(line[0][1] != line[0][3]) or ( i != 0  and np.sort(np.abs(line[0][1]-lines_y))[0] < 100):
            continue
        lines_y.append(line[0][1])
        i = i + 1
        if i == 3:
            break
    Lines = np.sort(lines_y)[1:]

    mm = 20
    Gray = img[Lines[0]+mm:Lines[1]-mm, StartVertical:y-50]
    Neg = negThresh[Lines[0]+mm:Lines[1]-mm, StartVertical:y-50]
    

    hull = convex_hull_image(Neg)
    contours,_ = cv.findContours((hull).astype(np.uint8),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    x,y,w,h = cv.boundingRect(cnt)
    kk = 0
    Neg = Neg[y-kk:y+h+kk,x-kk:x+w+kk]
    Gray = Gray[y-kk:y+h+kk,x-kk:x+w+kk]
    Bin = np.max(Neg)-Neg
    
    return Gray, Bin, Neg


def SemiPreprocessing(img):
    _, y = img.shape

    blur = cv.GaussianBlur(img,(11,11),0)
    _,Thresh = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    negThresh = np.max(Thresh) - Thresh

    StartVertical = np.where(negThresh[0, :] == 0)[0][0] * 5
    lines = cv.HoughLinesP(negThresh, 1, math.pi/2, 3, 3, 200, 1)

    i = 0
    lines_y = []
    for line in lines:
        if(line[0][1] != line[0][3]) or ( i != 0  and np.sort(np.abs(line[0][1]-lines_y))[0] < 100):
            continue
        lines_y.append(line[0][1])
        i = i + 1
        if i == 3:
            break
    Lines = np.sort(lines_y)[1:]

    mm = 20
    Gray = img[Lines[0]+mm:Lines[1]-mm, StartVertical:y-50]
    Neg = negThresh[Lines[0]+mm:Lines[1]-mm, StartVertical:y-50]
    Bin = np.max(Neg)-Neg
    
    return Gray, Bin, Neg


def FastPreprocessing(img):
    if np.max(img)<=1:
        img = np.array(255*img,dtype = np.uint8)


    _,Bin = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    s = 900
    e = 2800
    if InkDensity(Bin) > 250:
        s = 400
        e = 2000
    Gray = img[s:e,:]
    Bin = Bin[s:e,:]
    (_, w) = np.shape(Gray)

    Bin[:, :50] = 255
    Bin[:, w - 50:] = 255
    Neg = np.max(Bin)-Bin
    return Gray, Bin, Neg


def Segmentation(Gray, Bin):
    GraySentences = []
    BinSentences = []

    HorizontalProjection =  np.sum(255-Bin, axis=1, dtype=int) // 255
#     bar( [i for i in range (len(HorizontalProjection))], HorizontalProjection, width=0.8, align='center')
    peaks2, _ = find_peaks(HorizontalProjection, prominence=1,distance=150) 

    m = 0
    for i in range(len(peaks2)-1):
        k = 0
        x = np.sum(Bin[peaks2[i]+k:peaks2[i+1]+k,:])
        if(x>2000000):
            GraySentences.append(Gray[peaks2[i]+k:peaks2[i+1]+k,:])
            BinSentences.append(Bin[peaks2[i]+k:peaks2[i+1]+k,:])
            m = m + 1

    return GraySentences, BinSentences

################################################################################################################



def predict(clf,X_test):
    return np.bincount(clf.predict(X_test)).argmax()


def adaboost_clf(Y_train, X_train, clfNum=3, numClassifiers=50, learnRate=1.5):

    classifiers = {
        1: DecisionTreeClassifier(max_depth=1, random_state=1),
        2: KNeighborsClassifier(n_neighbors=3),
        3: SVC(C=5.0, gamma='auto', probability=True)
    }
    clf = classifiers.get(clfNum, None)

    if (clfNum == 1):
        clf = AdaBoostClassifier(clf,n_estimators=numClassifiers,learning_rate=learnRate,algorithm="SAMME")

    clf.fit(X_train, Y_train)
    return clf

####################################################################################################

def getTexture(G,B):
    (h, w) = np.shape(G)

    my = 200
    mx = 50
    widths = []

    sy = my
    sx = mx
    h_total = 0
    n = 0
    line_spacing=1/2
    BS=(256, 256)
    TBS = []

    Sum =  255*np.ones((h, w), dtype=np.uint8)
    contours = cv.findContours(255 - B , cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

    for cnt in contours[::-1]:
        (x, y, dw, dh) = cv.boundingRect(cnt)
        if dw == 0:
            dw = w
        if dh == 0:
            dh = h

        if dw * dh > 200:
            mu = cv.moments(B[y:y + dh, x:x + dw])
            if mu['m00'] == 0:
                mu['m00'] = 0.001
            center = ((mu['m01'] / mu['m00']), (mu['m10'] / mu['m00']))
            by = sy 

            if by<0:
                continue

            Sum[by:by + dh, sx:sx + dw] = np.minimum(B[y:y + dh, x:x + dw],Sum[by:by + dh,sx:sx + dw])


            sx += dw
            h_total += dh
            n += 1

            if sx >= BS[1] + mx:
                widths.append(sx)
                sy = sy + int(line_spacing * h_total / n)
                sx = mx
                h_total = 0
                n = 0
                
                if sy >= BS[0] + my:
                    end_x = np.min(widths)
                    end_y = sy
                    sy = my
                    Sum2 = Sum[my:min(BS[0] + my, end_y), mx:min(BS[1] + mx, end_x)].copy()
                    TBS.append(Sum2)
                    Sum =  255*np.ones((h, w), dtype=np.uint8)
                    
    if TBS == []:
        TBS.append(Sum[my:BS[0] + my,mx:BS[1] + mx])

    return TBS

#####################################################################################################

def extract(G,B):
    features = []
    features.extend(LBPhistogram(G,B))

    return features

def LBPhistogram(G,B):
    hist = np.zeros(256)

    for i in range(len(B)):
        hist = getLBPhistogram(G[i], B[i], hist)

    hist /= np.mean(hist)

    return hist

def getLBPhistogram(image, mask, hist=None):
    h, w = image.shape

    LBP = np.zeros((h, w), dtype=np.uint8)

    x = [0, 3, 3, 3, 0, -3, -3, -3]
    y = [3, 3, 0, -3, -3, -3, 0, 3]

    for i in range(8):
        view_shf = shift(image, (y[i], x[i]))
        view_image = shift(image, (-y[i], -x[i]))
        view_LBP = shift(LBP, (-y[i], -x[i]))
        res = (view_image >= view_shf)
        view_LBP |= (res.view(np.uint8) << i)

    hist = cv.calcHist([LBP], [0], mask, [256], [0, 256], hist, True)
    hist = hist.ravel()

    return hist

def shift(image, s) -> np.ndarray:
    r, c = s[0], s[1]

    if r >= 0:
        R = image[r:, :]
    else:
        R = image[0:r, :]

    if c >= 0:
        R = R[:, c:]
    else:
        R = R[:, 0:c]

    return R