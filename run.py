import datetime
import time
import orange, orngSVM
import numpy as np
import Orange
import multiprocessing
from multiprocessing import Pool, Lock
import numpy as np
import os
from Goldenberry.feature_selection.WKiera import GbWrapperCostFunction
from Goldenberry.classification.base.GbFactory import GbFactory
from Goldenberry.optimization.edas.Bivariate import Bmda, DependencyMethod

# Experiment Parameters
results_path = 'results.tab'
train_size = 800
pop = 20
gens = 80
runs = 10
noise_options = [5, 10]
take_files = 5
lock = Lock()

def run_single_exp(args):
    [path, noise_size] = args;
    file_name = os.path.basename(path)

    print '[', file_name, noise_size ,'] running ...', datetime.datetime.utcnow()
    tic = time.time()
    results = run_exp(path, noise_size)
    elapsed = time.time() - tic;   
    print '[Done]', file_name, noise_size, elapsed, 'seconds'
    
    with lock:
        save_to_file(results_path, '"' + file_name + '"', noise_size, elapsed, results)
    
def run_exp(file_path, noise_size):
    
    original_data = Orange.data.Table(file_path)
    data, relevant, noisy = add_uniform_noise(original_data, noise_size)
    r_scores = run_relieff(data, relevant, noisy)
    k_scores = run_kiedra(data, relevant, noisy)
    
    return [relevant, noisy, r_scores, k_scores]
    
def setup_kiedra(data):
    
    train_data, test_data = sample_data(data, train_size)
    svm_params = {
        'svm_type': orngSVM.SVMLearner.C_SVC,
        'kernel_type': orngSVM.SVMLearner.RBF,
        'C': 100,
        'gamma': 10,
        'normalization': False
    }
    factory = GbFactory(orngSVM.SVMLearner, svm_params)
    cost_func = GbWrapperCostFunction(train_data, test_data, factory, 0, 2, 2)
    bmda = Bmda()
    bmda.setup(pop, gens, dependency_method = DependencyMethod.sim, independence_threshold = 0.1)
    bmda.cost_func = cost_func
    return bmda

def create_noisy_variables(num):
    return [orange.FloatVariable('R' + str(i)) for i in range(1, num + 1)]

def add_uniform_noise(data, n):
    noisy_variables = create_noisy_variables(n)
    noisy_names = [a.name for a in noisy_variables]
    relevant_names = [a.name for a in data.domain.features]
    npdata, np_class = data.to_numpy('a')[0], data.to_numpy('c')[0]
    length, _ = npdata.shape
    noise_data = np.random.rand(length, len(noisy_variables)) * 10
    np_new_data = np.concatenate((npdata, noise_data, np_class), axis=1)
    new_domain = orange.Domain(data.domain.attributes + noisy_variables + [data.domain.class_var])
    return Orange.data.Table(new_domain, np_new_data), relevant_names, noisy_names

def run_relieff(data, relevant_variables, noisy_variables):
    relieff = Orange.feature.scoring.Relief()
    relevant_variables = data.domain.features
    scores = [ relieff(a, data) for a in data.domain.attributes ]
    return scores

def run_kiedra(data, relevant_variables, noisy_variables):
    runs_results = np.zeros(len(data.domain.attributes))
    for i in range(runs):
        kiedra = setup_kiedra(data)    
        result = kiedra.search()
        runs_results += result.params

    scores = runs_results / float(runs) 
    return scores.tolist()

def sample_data(data, n):
    sampler = Orange.data.sample.SubsetIndices2(p0=n)
    ind = sampler(data)
    traindata = data.select(ind, 0)
    testdata = data.select(ind, 1)
    traindata.shuffle()
    testdata.shuffle()
    return traindata, testdata

def files(rootPath):
    found_files = []
    for (path, dirs, files) in os.walk(rootPath):
        for file_name in files:
            if '.tab' not in file_name:
                continue
            
            found_files.append(path + '\\' +  file_name)
    return found_files

def clean_results():
    
    if os.path.exists(results_path):
        os.remove(results_path)

    with open(results_path, 'a') as file:
        file.write('\t'.join(['FILE', 'NOISE', 'ELAPSED(s)', 'RELEVANT', 'NOISY' 'R_SCORES', 'K_SCORES']) + '\n')

def save_to_file(path, file_name, noise_size, elapsed, results):
    data = [file_name, noise_size, elapsed] + results
    result_string = '\t'.join([str(val) for val in data]) + '\n'
    with open(path, 'a') as file:
        file.write(result_string)

def run():    
    clean_results()
    experiments = []
    pool = Pool(multiprocessing.cpu_count())
    
    exp_files = files('../model-free-data/3way/')[:take_files] + \
        files('../model-free-data/4way/')[:take_files] + \
        files('../model-free-data/4wayNoLow/')[:take_files] + \
        files('../model-free-data/5way/')[:take_files] + \
        files('../model-free-data/5wayNoLow/')[:take_files]
    for path in exp_files:
        for n in noise_options:
            experiments.append([path, n])

    results = pool.map(run_single_exp, experiments)

    #run_single_exp(['../model-free-data/5way/5way-best201.tab', 5])  

if __name__ == '__main__':
    multiprocessing.freeze_support()
    run()