import orange, orngSVM
import Orange
import multiprocessing
from multiprocessing import Pool, Lock
import numpy as np
import os
from Goldenberry.feature_selection.WKiera import GbWrapperCostFunction
from Goldenberry.classification.base.GbFactory import GbFactory
from Goldenberry.optimization.edas.Bivariate import Bmda, DependencyMethod

lock = Lock()
results_path = 'results.tab'

def run_single_exp(args):
    [path, noise_size] = args
    file_name = os.path.basename(path)

    print '[', file_name, noise_size ,'] running ...'
    result = run_exp(path, noise_size)   
    print '[Done]', file_name, noise_size
    
    with lock:
        save_to_file(results_path, file_name, result)
    
def run_exp(file_path, noise_size):
    
    data = Orange.data.Table(file_path)
    relevant_variables = data.domain.features
    data = add_uniform_noise(data, noise_size)
    
    relieff = Orange.feature.scoring.Relief()
    relief_relevant = relief_get_relevant_variables(relieff, data)
    relieff_has_found = has_found_only_relevants(relevant_variables, relief_relevant)
    
    bmda = setup_bmda(data)
    bmda_relevant = bmda_get_relevant_variables(bmda, data)
    bmda_has_found = has_found_only_relevants(relevant_variables, bmda_relevant)
    
    return [relieff_has_found, relief_relevant, bmda_has_found, bmda_relevant]

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

def relief_get_relevant_variables(score, data):
    relevant_attrs = [ a.name for a in data.domain.attributes if score(a, data) > 0.0]
    return relevant_attrs

def bmda_get_relevant_variables(bmda, data):
    result = bmda.search()
    relevant_attrs = [a.name for idx, a in enumerate(data.domain.attributes) if result.params[idx]]
    return relevant_attrs

def sample_data(data, n):
    sampler = Orange.data.sample.SubsetIndices2(p0=n)
    ind = sampler(data)
    traindata = data.select(ind, 0)
    testdata = data.select(ind, 1)
    traindata.shuffle()
    testdata.shuffle()
    return traindata, testdata

def has_found_only_relevants(relevant_variables, found_var_names):
    all_varaibles_found = all(relevant.name in found_var_names for relevant in relevant_variables)
    found = (len(relevant_variables) == len(found_var_names)) and all_varaibles_found
    return found

def get_file_paths():
    for (path, dirs, files) in os.walk('../model-free-data'):
        for file_name in files:
            if '.tab' not in file_name:
                continue
            
            yield path + '\\' +  file_name

def save_to_file(path, file_name, result):
    result_string = file_name + '\t'.join([str(val) for val in result]) + '\n'
    with open(path, 'a') as file:
        file.write(result_string)

def add_header():
    with open(results_path, 'a') as file:
        file.write('\t'.join(['FILE', 'RELIEF_WORKED', 'RELIEF_VARIABLES', 'BMDA_WORKED', 'BMDA_VARIABLES']) + '\n')
def run():
    if os.path.exists(results_path):
        os.remove(results_path)

    add_header()
    experiments = []
    pool = Pool(multiprocessing.cpu_count())
    for path in get_file_paths():
        for n in [5, 10, 20]:
            experiments.append([path, n])

    results = pool.map(run_single_exp, experiments)  

if __name__ == '__main__':
    multiprocessing.freeze_support()
    run()