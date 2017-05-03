import os
os.chdir('/Users/benchan/Desktop/Python data')

#Load Libararies
from pandas import read_csv
from pandas.tools.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Load dataset
filename = 'iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(filename, names=names)
print(dataset.head(20))

#19.3 Summarize the dataset
#19.3.1 Dimensions of Dataset
#shape
print(dataset.shape)

#19.3.2 Peek at the data
#head
print(dataset.head(20))

#19.3.3 statistical summary
#description
print(dataset.describe())
#we can see that all of the numerical values have the same scale (centimeters) and similar ranges between 0 and 8 cm
#standard deviation - how spread out the data is
#deviation - how far from normal

#19.3.4 Class Distribution
#class distribution
print(dataset.groupby('class').size())
#see that each class has the same number of instances 50 or ~33% of the dataset

#19.4 Data Visualization
#univariate plots to better understand each attribute
#multivariate plots to better understand the relationship between attributes

#19.4.1 Univariate Plots
#given that the input variables are numeric
#box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

#histograms
dataset.hist()
pyplot.show()
#look like perhaps two of the input variables have a gaussian distribution 'sepal length & sepal width'

#19.4.2 Multivariate plots
#help spot structured relationship between variables
#scatter plot matrix
scatter_matrix(dataset)
pyplot.show()
#note the diagonal grouping of some paris of attributes. this suggests a high correlation and a predictable 
#relationship

#19.5.1 create a validation dataset
#We need to know whether or not the model that we created is any good. Later, we will use
#statistical methods to estimate the accuracy of the models that we create on unseen data.
#We also want a more concrete estimate of the accuracy of the best model on unseen data by
#evaluating it on actual unseen data. That is, we are going to hold back some data that the
#algorithms will not get to see and we will use this data to get a second and independent idea of
#how accurate the best model might actually be. We will split the loaded dataset into two, 80%
#of which we will use to train our models and 20% that we will hold back as a validation dataset.

#Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size= 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = validation_size, random_state=seed)

#test harness
#We will use 10-fold cross-validation to estimate accuracy. This will split our dataset into 10
#parts, train on 9 and test on 1 and repeat for all combinations of train-test splits. We are using
#the metric of accuracy to evaluate models. This is a ratio of the number of correctly predicted
#instances divided by the total number of instances in the dataset multiplied by 100 to give a
#percentage (e.g. 95% accurate). We will be using the scoring variable when we run build and
#evaluate each model next.

#19.5.3 Build Models
#We don't know which algorithms would be good on this problem or what configurations to use.
#We get an idea from the plots that some of the classes are partially linearly separable in some
#dimensions, so we are expecting generally good results. Let's evaluate six different algorithms:

#This list is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB
#and SVM) algorithms. We reset the random number seed before each run to ensure that the
#evaluation of each algorithm is performed using exactly the same data splits. It ensures the
#results are directly comparable. Let's build and evaluate our five models:

#spot-check algortihm
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
#evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train,cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
#we now have 6 models and accuracy estimation for each.
#We can see that it looks like KNN has the largest estimated accuracy score. We can also
#create a plot of the model evaluation results and compare the spread and the mean accuracy
#of each model. There is a population of accuracy measures for each algorithm because each
#algorithm was evaluated 10 times (10 fold cross-validation).

#We can see that it looks like KNN has the largest estimated accuracy score. We can also
#create a plot of the model evaluation results and compare the spread and the mean accuracy
#of each model. There is a population of accuracy measures for each algorithm because each
#algorithm was evaluated 10 times (10 fold cross-validation).

#compare Algorithm
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
#you can see that the box and whisker plots are squashed at the top of the range, with many samples achieveing 
#100% accuracy

#19.6 Make prediction
#The KNN algorithm was the most accurate model that we tested. Now we want to get an idea
#of the accuracy of the model on our validation dataset. This will give us an independent final
#check on the accuracy of the best model. It is important to keep a validation set just in case
#you made a slip during training, such as overfitting to the training set or a data leak. Both
#will result in an overly optimistic result. We can run the KNN model directly on the validation
#set and summarize the results as a final accuracy score, a confusion matrix and a classification
#report.

#make prediction on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
prediction = knn.predict(X_validation)
print(accuracy_score(Y_validation, prediction))
print(confusion_matrix(Y_validation, prediction))
print(classification_report(Y_validation, prediction))

#we can see that the accuracy is 90%. the confusion matrix provides an indication of the three errors made
#FInally the classification report provides a breakdown of each class by precision, recall, f1 score and support
#showing excellent results (granted the validation dataset was small)

#The f1-score gives you the harmonic mean of precision and recall. The scores corresponding to every class
#will tell you the accuracy of the classifier in classifying the data points in that particular class compared
#to all other classes.

#The support is the number of samples of the true response that lie in that class.
