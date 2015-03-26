import csv
import numpy
import scipy
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn import neighbors
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import knnplots

from sklearn.naive_bayes import GaussianNB

from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV

filename = "wdbc.csv"
fileOpen = open(filename, "rU")
csvData = csv.reader(fileOpen)
dataList = list(csvData)
dataArray = numpy.array(dataList)

X = dataArray[:,2:32].astype(float)# index out rest of the data
y = dataArray[:, 1] # diagnosis column
yFreq = scipy.stats.itemfreq(y)
print yFreq

#.bar(left = 0, height = int(yFreq[0][1]), color = "red")
#plt.bar(left = 1, height = int(yFreq[1][1]))

# convert 'M' and 'B' to 1 and 0
le = preprocessing.LabelEncoder()
le.fit(y)
yTransformed = le.transform(y)
#plt.legend(loc = "upper right")
#plt.show()

# construct a correlation matrix between columns
correlationMatrix = numpy.corrcoef(X, rowvar = 0)
fig, ax = plt.subplots()
#heatmap = ax.pcolor(correlationMatrix, cmap = plt.cm.Reds)
#plt.show()

#plt.scatter(x = X[:,0], y = X[:,1], c=y)
#plt.show()

def scatter_plot(X,y):
    # set the siz eof the figure
    plt.figure(figsize = (2*X.shape[1],2*X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):

            plt.subplot(X.shape[1],X.shape[1],i+1+j*X.shape[1])
            if i == j:
                plt.hist(X[:, i][y=="M"], alpha = 0.4, color = 'm',
                  bins = numpy.linspace(min(X[:, i]),max(X[:, i]),30))
                plt.hist(X[:, i][y=="B"], alpha = 0.4, color = 'b',
                  bins = numpy.linspace(min(X[:, i]),max(X[:, i]),30))
                plt.xlabel(i)
            else:
                plt.gca().scatter(X[:, i], X[:, j],c=y, alpha = 0.4)
                plt.xlabel(i)
                plt.ylabel(j)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
    plt.tight_layout()
    #plt.show()
    #plt.subplot

#scatter_plot(X[:,:5],y)

# compute the two nearest neighbours of each element in X
nbrs = neighbors.NearestNeighbors(n_neighbors=3, algorithm="ball_tree").fit(X)
distances, indices = nbrs.kneighbors(X)

#Train the classification based on 3 nearest neighbours
knnK3 = neighbors.KNeighborsClassifier(n_neighbors=3)
knnK3 = knnK3.fit(X, yTransformed)
predictedK3 = knnK3.predict(X)

#Train the classification based on 15 nearest neighbours
knnK15 = neighbors.KNeighborsClassifier(n_neighbors=15)
knnK15= knnK15.fit(X, yTransformed)
predictedK15 = knnK15.predict(X)


# Taking into account the distances between the points (weights)
knnWD = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
knnWD = knnWD.fit(X, yTransformed)
predictedWD = knnWD.predict(X)

nonAgreement = predictedK3[predictedK3 != predictedWD]
#print 'Number of discrepencies', len(nonAgreement)

#splits training dataset 25% for testing
XTrain, XTest, yTrain, yTest = train_test_split(X, yTransformed)

# make a confusion matrix
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(XTrain, yTrain)                             #classify the training set
predicted = knn.predict(XTest)                      # predict the class labels for the test set

mat = metrics.confusion_matrix(yTest, predicted)    # Compute the confusion matrix for our predicitons
print mat
print metrics.classification_report(yTest, predicted)
print "accuracy: ", metrics.accuracy_score(yTest, predicted)

# Plot accuracy score for different numbers of neighbours and weights

knnplots.plotaccuracy(XTrain,yTrain,XTest,yTest,310)
knnplots.decisionplot(XTrain, yTrain, n_neighbors=3,weights="uniform")

# k- fold validation splits the training datasets into several different test sets

knn3scores = cross_validation.cross_val_score(knnK3, XTrain, yTrain, cv = 5)
print knn3scores
print "Mean f scores KNN3", knn3scores.mean()
print "SD of scores KNN3", knn3scores.std()

knn15scores = cross_validation.cross_val_score(knnK15, XTrain, yTrain, cv = 5)
print knn15scores
print "Mean f scores KNN15", knn15scores.mean()
print "SD of scores KNN15", knn15scores.std()

# Plot accuracy means and standard deviations for different numbers of folds

meansKNNK3 = []
sdsKNNK3 = []
meansKNNK15 = []
sdsKNNK15 = []

ks = range (2, 21)

for k in ks:
  knn3scores = cross_validation.cross_val_score(knnK3, XTrain,
                                                yTrain, cv=k)
  knn15scores = cross_validation.cross_val_score(knnK15, XTrain,
                                                yTrain, cv=k)
  meansKNNK3.append(knn3scores.mean())
  sdsKNNK3.append(knn3scores.std())
  meansKNNK15.append(knn15scores.mean())
  sdsKNNK15.append(knn15scores.std())

plt.plot(ks, meansKNNK3, label="KNN 3 mean accuracy", color="purple")
plt.plot(ks, meansKNNK15, label="KNN 15 mean accuracy", color="yellow")
plt.legend(loc=3)
plt.ylim(0.5,1)
plt.title("Accuracy mean with increasing k")
plt.show()

plt.plot(ks, sdsKNNK3, label="KNN 3 sd accuracy", color="purple")
plt.plot(ks, sdsKNNK15, label="KNN 15 sd accuracy",color="yellow")
plt.legend(loc=3)
plt.ylim(0, 0.1)
plt.title("Accuracy standard deviations with Increasing K")
plt.show()