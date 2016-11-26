import orange, orngSVM
import Orange
import numpy as np
from Goldenberry.feature_selection.WKiera import GbWrapperCostFunction
from Goldenberry.classification.base.GbFactory import GbFactory
from Goldenberry.optimization.edas.Bivariate import Bmda, DependencyMethod

def run():
    data = Orange.data.Table(r'C:\Python27\Lib\site-packages\Orange\datasets\genomic-datasets\5way\5way-best201')
    data = add_uniform_noise(data, 5)
    bmda = setup_bmda(data)
    relieff = Orange.feature.scoring.Relief()
    result = bmda.search()
    run_relieff(relieff, data)    

def setup_bmda(data):
    train_data, _ = sample_data(data, 400)
    svm_params = {
        'svm_type': orngSVM.SVMLearner.C_SVC,
        'kernel_type': orngSVM.SVMLearner.RBF,
        'C': 100,
        'gamma': 10,
        'normalization': False
    }
    
    factory = GbFactory(orngSVM.SVMLearner, svm_params)
    cost_func = GbWrapperCostFunction(train_data, data, factory, 0, 4, 2)
    bmda = Bmda()
    bmda.setup(40, 120, dependency_method = DependencyMethod.sim, independence_threshold = 0.1)
    bmda.cost_func = cost_func
    return bmda

def create_noise_features(num):
    return [orange.FloatVariable('R' + str(i)) for i in range(1, num + 1)]

def add_uniform_noise(data, n):
    noise_features = create_noise_features(n)
    npdata, np_class = data.to_numpy('a')[0], data.to_numpy('c')[0]
    length, _ = npdata.shape
    noise_data = np.random.rand(length, len(noise_features)) * 10
    np_new_data = np.concatenate((npdata, noise_data, np_class), axis=1)
    new_domain =  orange.Domain(data.domain.attributes + noise_features + [data.domain.class_var])
    return Orange.data.Table(new_domain, np_new_data)

def run_relieff(score, data):
    aux = [ (a.name, score(a, data)) for a in data.domain.attributes]
    print aux

def sample_data(data, n):
    sampler = Orange.data.sample.SubsetIndices2(p0=n)
    ind = sampler(data)
    traindata = data.select(ind, 0)
    testdata = data.select(ind, 1)
    print 'data: ', len(traindata), len(testdata)
    traindata.shuffle()
    testdata.shuffle()
    return traindata, testdata

run()