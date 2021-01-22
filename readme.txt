You should only run main.py
To Try different things you should only change the last line


# Approach: 1= Lines LBP, 2= Texture Blocks LBP, 3= 6 features
# Classifier: 1= Adaboosted Decision Trees, 2= 3-NN, 3= SVM
# PreprocessingNum: 0= FastPreprocessing, 1= SemiPreprocessing, 2= AccuratePreprocessing
# dataSet: Folder containing Folders to Test (Relative to this file path)
# outTxt: The results.txt file
# timesTxt: The time of each Sample time.txt

## Our choosend method is line LBP using Fast Preprocessing with SVM
Run(Approach=1,Classifier=3,PreprocessingNum=0,dataSet="data",outTxt="results.txt",timesTxt="time.txt")


################################################################
These two methods are used to generate Test samples
The first one sorts the database and but each writer Samples in a folder with his ID
The second generate N Test Sample to test:

SortDataset(formFile,datasetPath,outPath)
GenerateNSamples(N,sortedDataset,outSamplePath)