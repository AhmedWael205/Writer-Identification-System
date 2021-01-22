from commonFunctions import *

formFile = 'D:/University Materials/CMPN courses/Pattern Recognition/Project/forms.txt'
datasetPath = "D:/University Materials/CMPN courses/Pattern Recognition/Project/Dataset/"
outPath = "D:/University Materials/CMPN courses/Pattern Recognition/Project/Sorted/"
# SortDataset(formFile,datasetPath,outPath)

sortedDataset = outPath
outSamplePath = "D:/University Materials/CMPN courses/Pattern Recognition/Project/data"
N = 200
# GenerateNSamples(N,sortedDataset,outSamplePath)


def Run(Approach=1,Classifier=3,PreprocessingNum=0,dataSet="data",outTxt="results.txt",timesTxt="time.txt",Accuarcy=False):

    NumberOfWriterPerTest = 3
    NumberOfWirterSamples = 2

    writerIDs = range(1, NumberOfWriterPerTest+1)

    Tests = (glob.glob(dataSet+"/*/"))
    Tests = sorted(Tests,key=lambda x: int((x).split("\\")[1]))

    myfile = open(outTxt, "w")
    mytime = open(timesTxt, "w")

    times = []

    TestNum = 0
    
    if Approach == 1:
        print("Lines LBP")
        for Test in Tests:
            try:
                t1 = time.time()
                features, labels = [], []
                for writer in writerIDs:
                    total_gray_lines, total_bin_lines = [], []

                    for i in range(NumberOfWirterSamples):

                        pathSample = Test+str(writer)+"/"+str(i+1)+".png"
                        gray = cv.imread(pathSample,0)
                        gray_img,_ ,bin_img= Preprocessing(gray,PreprocessingNum)
                        gray_lines, bin_lines = Segmentation(gray_img, bin_img)
                        total_gray_lines.extend(gray_lines)
                        total_bin_lines.extend(bin_lines)

                    # Extract features of every line separately
                    x, y = [], []
                    for g, b in zip(total_gray_lines, total_bin_lines):
                        f = extract([g], [b])
                        x.append(f)
                        y.append(writer)

                    features.extend(x)
                    labels.extend(y)

                clf = adaboost_clf(labels, features, clfNum=Classifier)
                pathTest = Test+"test.png"

                total_gray_lines, total_bin_lines = [], []

                gray = cv.imread(pathTest,0)
                gray_img,_ ,bin_img= Preprocessing(gray,PreprocessingNum)
                gray_lines, bin_lines = Segmentation(gray_img, bin_img)
                total_gray_lines.extend(gray_lines)
                total_bin_lines.extend(bin_lines)

                # Extract features of every line separately
                x = []
                for g, b in zip(total_gray_lines, total_bin_lines):
                    f = extract([g], [b])
                    x.append(f)

                # Get the most likely writer
                r = predict(clf,x)
                t2 = round((time.time() - t1),2)
                times.append(t2)
                mytime.write(str(t2)+"\n")
                myfile.write(str(r)+"\n")
                print(str(TestNum+1),": ",str(r))
                TestNum+=1

            except:
                print("Error at " + Test)
                t2 = round((time.time() - t1),2)
                times.append(t2)
                mytime.write(str(t2)+"\n")
                myfile.write(str(2)+"\n")
                print(str(TestNum+1),": ",str(2))
                TestNum+=1
                
    elif Approach == 2:
        print("Texture Blocks LBP")
        for Test in Tests:
            try:
                t1 = time.time()
                features, labels = [], []
                for writer in writerIDs:
                    textureBlocks_all, masks = [], []
                    for i in range(NumberOfWirterSamples):

                        pathSample = Test+str(writer)+"/"+str(i+1)+".png"
                        gray = io.imread(pathSample)
                        G, B, N= Preprocessing(gray,PreprocessingNum)
                        textureBlocks = getTexture(G,B)
                        for c in textureBlocks:
                            masks.append(255-c)

                        textureBlocks_all.extend(textureBlocks)

                    # Extract features of every line separately
                    x, y = [], []
                    for g, b in zip(textureBlocks_all, masks):
                        f = extract([g], [b])
                        x.append(f)
                        y.append(writer)

                    features.extend(x)
                    labels.extend(y)

                clf = adaboost_clf(labels, features, clfNum=Classifier)

                pathTest = Test+"test.png"

                textureBlocks, masks = [], []

                gray = io.imread(pathTest)
                G, B, N= Preprocessing(gray,PreprocessingNum)
                textureBlocks = getTexture(G,B)
                for c in textureBlocks:
                    masks.append(255-c)

                # Extract features of every line separately
                x = []
                for g, b in zip(textureBlocks, masks):
                    f = extract([g], [b])
                    x.append(f)

                # Get the most likely writer
                r = predict(clf,x)
                t2 = round((time.time() - t1),2)
                times.append(t2)
                mytime.write(str(t2)+"\n")
                myfile.write(str(r)+"\n")
                print(str(TestNum+1),": ",str(r))
                TestNum+=1
            except:
                print("Error at " + Test)
                t2 = round((time.time() - t1),2)
                times.append(t2)
                mytime.write(str(t2)+"\n")
                myfile.write(str(2)+"\n")
                print(str(TestNum+1),": ",str(2))
                TestNum+=1
                
    else:
        print("6 Features")
        for Test in Tests:
            try:
                t1 = time.time()
                features, labels = [], []
                for writer in writerIDs:
                    total_gray_lines, total_bin_lines = [], []
                    for i in range(NumberOfWirterSamples):

                        pathSample = Test+str(writer)+"/"+str(i+1)+".png"
                        gray = cv.imread(pathSample,0)
                        gray_img, _,bin_img = Preprocessing(gray,PreprocessingNum)
                        gray_lines, bin_lines = Segmentation(gray_img, bin_img)
                        f = np.asarray(FeatureExtraction(bin_lines, bin_img))

                        x,y = [],[]
                        for ff in f:
                            x.append(ff)
                            y.append(writer)
                        features.extend(x)
                        labels.extend(y)


                clf = adaboost_clf(labels, features, clfNum=Classifier)
                pathTest = Test+"test.png"

                total_gray_lines, total_bin_lines = [], []

                gray = cv.imread(pathTest,0)
                gray_img, _,bin_img = Preprocessing(gray,PreprocessingNum)
                gray_lines, bin_lines = Segmentation(gray_img, bin_img)
                f = np.asarray(FeatureExtraction(bin_lines, bin_img))
                x= []
                for ff in f:
                    x.append(ff)

                r = predict(clf,x)
                t2 = round((time.time() - t1),2)
                times.append(t2)
                mytime.write(str(t2)+"\n")
                myfile.write(str(r)+"\n")
                print(str(TestNum+1),": ",str(r))
                TestNum+=1
            except:
                print("Error at " + Test)
                t2 = round((time.time() - t1),2)
                times.append(t2)
                mytime.write(str(t2)+"\n")
                myfile.write(str(2)+"\n")
                print(str(TestNum+1),": ",str(2))
                TestNum+=1
        

    
    myfile.close()
    mytime.close()
    if Accuarcy:
        calcAccuarcy(outTxt,dataSet+"/out.txt")
    print("Average Time: "+str(round(np.mean(np.asarray(times)),2)))


##################################################################################################

# Approach: 1= Lines LBP, 2= Texture Blocks LBP, 3= 6 features
# Classifier: 1= Adaboosted Decision Trees, 2= 3-NN, 3= SVM
# PreprocessingNum: 0= FastPreprocessing, 1= SemiPreprocessing, 2= AccuratePreprocessing
# dataSet: Folder containing Folders to Test (Relative to this file path)
# outTxt: The results.txt file
# timesTxt: The time of each Sample time.txt

## Our choosend method is line LBP using Fast Preprocessing with SVM
Run(Approach=1,Classifier=3,PreprocessingNum=0,dataSet="data",outTxt="results.txt",timesTxt="time.txt")