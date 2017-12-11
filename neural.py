from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

def getArray(data):
    array = []
    for line in data:
        line = line.strip('\n')
        values = line.split(',')
        item = []
        for value in values:
            item.append(float(value))
        array.append(item)
    return array

def prepareData(data):
    array = getArray(data)
    features = []
    target = []
    for item in array:
        features.append(item[:-1])
        target.append(item[8])
    return features, target

# def getPrecision(
#         clf,
#         X_train,
#         X_test,
#         y_train,
#         y_test
# ):
#     clf.fit(X_train, y_train)
#     prediction = clf.predict(X_test)
#     right = 0
#     wrong = 0
#     for i in range(len(y_test)):
#         if y_test[i] == prediction[i]:
#             right += 1
#         else:
#             wrong += 1
#     return right / len(y_test) * 100

def getCorrelation(
        clf,
        X_train,
        X_test,
        y_train,
        y_test
):
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    return pearsonr(prediction, y_test)[0]

def getActivation(i):
    values = ['identity', 'logistic', 'tanh', 'relu']
    return values[i]

def getSolver(i):
    values = ['lbfgs', 'sgd', 'adam']
    return values[i]

def run(
        hidden_layer_sizes,
        activation,
        solver,
        alpha,
        learning_rate,
        momentum
):

    f = open('Concrete_Data.csv', 'r')
    features, target = prepareData(f)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.30)

    clf = MLPRegressor(
        # The ith element represents the number of neurons in the ith hidden layer.
        hidden_layer_sizes=hidden_layer_sizes,
        # Activation function for the hidden layer.
        activation=getActivation(activation),
        # The solver for weight optimization.
        solver=getSolver(solver),
        # L2 penalty (regularization term) parameter.
        alpha=alpha,
        # The initial learning rate used.
        learning_rate_init=learning_rate,
        # Momentum for gradient descent update. Should be between 0 and 1.
        momentum=momentum,
        max_iter=5000
    )
    # return getPrecision(clf, X_train, X_test, y_train, y_test)
    return getCorrelation(clf, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    print(
        'correlation',
        run(
            (5, 2),
            1,
            1,
            0.0001,
            0.001,
            0.9
        )
    )
