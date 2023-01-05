import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import seaborn as sb
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE


def importData():
    df = pd.read_excel(r'Concrete_Data.xls')
    pd.set_option('display.max_columns', None)
    print("Dataframe description:")
    print(df.describe())
    print("Dataframe head:")
    print(df.head())
    data = df.iloc[1:, 0:-1]
    target = df.iloc[1:, -1]

    return data, target, df


def analizeData(target, df, data_np):
    print("Value counts:")
    print(target.value_counts())
    sb.displot(target)
    plt.savefig("plots/target_distribution.png")
    plt.show()
    stats.probplot(target,
                   plot=plt) # Note: The target variable is not normally distributed, but it is close to normal distribution
    plt.savefig("plots/probplot.png")
    plt.show()
    targetLog = np.log(target)
    stats.probplot(targetLog, plot=plt)  # Note: log transformation seems to be worse
    plt.savefig("plots/log_probplot.png")
    plt.show()
    sb.pairplot(df)
    plt.savefig("plots/pairplot.png")
    plt.show()
    print("HeatMap")
    createHeatMap(df)
    print("Number of NULL values in each column:")
    print(df.isnull().sum())
    plt.boxplot(data_np)
    plt.savefig("plots/boxplot.png")
    plt.show()
    print('---Mean---')
    print(data_np.mean(axis=0))
    print('--std---')
    print(data_np.std(axis=0))


def createHeatMap(df):
    correlation_matrix = df.corr().round(2)
    print("Correlation matrix:")
    print(correlation_matrix)
    sb.heatmap(data=correlation_matrix,
               annot=True)  # Note: The strongest correlation is between cement and compressive strength
    # Note: the heatmap confirms the results from the pairplot
    plt.savefig("plots/heatmap.png", bbox_inches='tight')
    plt.show()


def convertToNpArray(data, target):
    data_np = np.array(data, dtype=np.int16)
    target_np = np.array(target, dtype=np.int16)

    print(data_np.shape)
    print(target_np.shape)
    return data_np, target_np


def scaleData(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    print(scaled_data[1, :])
    print('---Mean---')
    print(scaled_data.mean(axis=0))
    print('--std---')
    print(scaled_data.std(axis=0))
    return scaled_data


def splitData(data, target, size):
    train_data, test_data, \
    train_target, test_target = \
        train_test_split(data, target, test_size=size, random_state=10)

    print("Training dataset:")
    print("train_data:", train_data.shape)
    print("train_target:", train_target.shape)
    print("Testing dataset:")
    print("test_data:", test_data.shape)
    print("test_target:", test_target.shape)
    return train_data, test_data, train_target, test_target


def trainLinearRegressionModel(train_data, train_target):
    linear_regression = LinearRegression()
    linear_regression.fit(train_data, train_target)
    return linear_regression


def getPredictedValues(linear_regression, test_data, test_target):
    realValues = []
    predictedValues = []
    ids = []
    for x in range(0, len(test_data)):
        prediction = linear_regression.predict(test_data[x, :].reshape(1, -1))
        realValues.append(test_target[x])
        predictedValues.append(prediction)
        ids.append(x)

    return realValues, predictedValues, ids


def createPlot(arr1, arr2, title):
    plt.scatter(arr1['ids'], arr1['arr'], label=arr1['label'])
    plt.scatter(arr2['ids'], arr2['arr'], label=arr2['label'])
    plt.title(title)
    plt.xlabel('ID')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/" + title + ".png")
    plt.show()


def prepDict(arr, ids, label):
    return {
        'arr': arr,
        'ids': ids,
        'label': label
    }


def crossValidate(data, target, cv):
    scores = cross_val_score(LinearRegression(), data, target, cv=cv)
    print(scores)
    print(scores.mean())
    return scores.mean()


def polynomialFeatures(trainData, testData, trainTarget, degree):
    poly = PolynomialFeatures(degree)
    trainDataPoly = poly.fit_transform(trainData)
    testDataPoly = poly.fit_transform(testData)
    lr_poly = LinearRegression()
    lr_poly.fit(trainDataPoly, trainTarget)
    return lr_poly, trainDataPoly, testDataPoly


def reduceFeatures(trainData, testData, trainTarget, degree,
                   numOfFeatures):  # Note: degree = 1 -> without polynomial features
    lr_poly, trainDataPoly, testDataPoly = polynomialFeatures(trainData, testData, trainTarget, degree)
    rfe = RFE(lr_poly, n_features_to_select=numOfFeatures)
    rfe = rfe.fit(trainDataPoly, trainTarget)
    print(rfe.support_)
    print(rfe.ranking_)
    return rfe, trainDataPoly, testDataPoly


def validateModel(data, target, model, name):
    realValues, predictedValues, ids = getPredictedValues(model, data, target)
    print("Mean squared error of a learned model: ", mean_squared_error(realValues, predictedValues))
    print('Variance score: ', r2_score(realValues, predictedValues))
    cv = []
    for x in range(2, 5):
        cv.append({
            'cv': x,
            'score': crossValidate(data, target, x)
        })
    realValsDict = prepDict(realValues, ids, "Real Values")
    predictedValsDict = prepDict(predictedValues, ids, "Predicted Values")
    createPlot(realValsDict, predictedValsDict, name)
    return {
        'mean': mean_squared_error(realValues, predictedValues),
        'variance': r2_score(realValues, predictedValues),
        'cv': cv
    }


def trainModels(data, target, postfix):
    dataSizes = [0.2, 0.3, 0.4, 0.5]

    results = []
    for size in dataSizes:
        models = []
        train_data, test_data, train_target, test_target = splitData(data, target, size)

        lrModel = trainLinearRegressionModel(train_data, train_target)
        models.append({
            'model': lrModel,
            'testData': test_data,
            'dataSize': size,
            'name': 'LR_s_' + str(size) + "_" + postfix,
            'type': 'LR'
        })

        for degree in range(2, 5):
            lrPolyModel, trainDataPoly, testDataPoly = polynomialFeatures(train_data, test_data, train_target, degree)
            models.append({
                'model': lrPolyModel,
                'testData': testDataPoly,
                'degree': degree,
                'dataSize': size,
                'name': 'PR_d_' + str(degree) + "_s_" + str(size) + "_" + postfix,  # Polynomial Regression Model
                'type': 'PR'
            })

        # Note: degree = 2 -> the best model for this dataset
        for numOfFeatures in range(5, 15):
            rfe, trainDataPoly, testDataPoly = reduceFeatures(train_data, test_data, train_target, 2, numOfFeatures)
            models.append({
                'model': rfe,
                'testData': testDataPoly,
                'numOfFeatures': numOfFeatures,
                'dataSize': size,
                'name': 'RFE_d_2_nf_' + str(numOfFeatures) + "_s_" + str(size) + "_" + postfix,
                'type': 'RFE'
            })

        for model in models:
            print(model['name'])
            results.append({
                'model': model,
                'validation': validateModel(model['testData'], test_target, model['model'], model['name'])
            })

    return results


def printModelsResults(results):
    for result in results:
        print(result['model']['name'])
        print("Mean squared error: ", result['validation']['mean'])
        print("Variance score: ", result['validation']['variance'])
        print("Cross Validation: ")
        for cv in result['validation']['cv']:
            print("CV: ", cv['cv'], " Score: ", cv['score'])
        print("")


def saveModelsResultsToCSV(results, filename):
    with open(filename, 'w') as csvfile:
        fieldnames = ['name', 'mean', 'variance', 'cv2', 'cv3', 'cv4']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                'name': result['model']['name'],
                'mean': result['validation']['mean'],
                'variance': result['validation']['variance'],
                'cv2': result['validation']['cv'][0]['score'],
                'cv3': result['validation']['cv'][1]['score'],
                'cv4': result['validation']['cv'][2]['score']
            })


def chooseBestModel(results):
    bestModel = results[0]
    for result in results:
        if result['validation']['mean'] >= bestModel['validation']['mean']:
            continue
        if result['validation']['variance'] <= bestModel['validation']['variance']:
            continue
        if result['validation']['cv'][0]['score'] <= 0:
            continue
        if result['validation']['cv'][1]['score'] <= 0:
            continue
        if result['validation']['cv'][2]['score'] <= 0:
            continue
        bestModel = result
    return bestModel


def getNBestModels(results, n):
    bestModels = []
    for i in range(0, n):
        bestModel = chooseBestModel(results)
        bestModels.append(bestModel)
        results.remove(bestModel)
    return bestModels


def getNBestLRModels(results, n):
    lrModels = []
    for result in results:
        if result['model']['type'] == 'LR':
            lrModels.append(result)
    bestModels = []
    for i in range(0, n):
        bestModel = chooseBestModel(lrModels)
        if bestModel['model']['type'] != 'LR':
            continue
        bestModels.append(bestModel)
        lrModels.remove(bestModel)
    return bestModels


def init():
    data, target, df = importData()
    data_np, target_np = convertToNpArray(data, target)
    analizeData(target, df, data_np)

    results = trainModels(data_np, target_np, 'ns')
    scaledResults = trainModels(scaleData(data_np), target_np, 's')

    print("Results:")
    print(results)
    print("Scaled results:")
    print(scaledResults)
    saveModelsResultsToCSV(results, 'non-scaled_results.csv')
    saveModelsResultsToCSV(scaledResults, 'scaled_results.csv')
    printModelsResults(results)
    printModelsResults(scaledResults)

    bestModel = chooseBestModel(results)
    print("Best model for non-scaled data: ", bestModel['model']['name'])
    printModelsResults([bestModel])

    bestModel = chooseBestModel(scaledResults)
    print("Best model for scaled data: ", bestModel['model']['name'])
    printModelsResults([bestModel])

    bestModelsScaled = getNBestModels(scaledResults, 3)
    print("Best models for scaled data: ")
    printModelsResults(bestModelsScaled)
    bestModelsNonScaled = getNBestModels(results, 3)
    print("Best models for non-scaled data: ")
    printModelsResults(bestModelsNonScaled)

    bestLRModelsScaled = getNBestLRModels(scaledResults, 3)
    print("Best LR models for scaled data: ")
    printModelsResults(bestLRModelsScaled)
    bestLRModelsNonScaled = getNBestLRModels(results, 3)
    print("Best LR models for non-scaled data: ")
    printModelsResults(bestLRModelsNonScaled)

init()
