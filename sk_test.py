from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.linear_model import LogisticRegression as lr
from sklearn.cross_validation import cross_val_score
from sklearn import svm
import numpy as np
def main():
    file = open('data_image', 'r')
    print file
    data = []
    for line in file:
        #print "Line:"
        #print line
        tmp = line.strip().split(' ')
        #print "Tmp:"
        #print tmp
        list = []
        for i in tmp:
            list.append(int(i))
        #print "List:"
        #list[-1] = int(list[-1])
        #print list
        data.append(list)
    [row, col] = [len(data), len(data[0])]
    #print data
    #print row,col
    data = np.array(data)
    #print type(data)
    print data.shape
    X = data[:,:-1]
    y = data[:,-1]
    print
    #model = rfc(n_estimators = 5000,oob_score=True, max_depth = None ,n_jobs = 16)
    #model = lr()
    model = svm.SVC()
    model.fit(X,y)
    '''
    idx = 0
    acc = 0
    ans = model.predict(X)
    for ele in ans:
        #ans = model.predict(ele)
        if ele == y[idx]:
           acc += 1
        idx += 1
    print acc
    print acc * 1.0 / row
    '''
    scores = cross_val_score(model, X, y)
    print scores.mean()
if __name__ == '__main__':
    main()