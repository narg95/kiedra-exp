import orange
import Orange
import numpy as np

def run():
    data = Orange.data.Table(r'C:\Python27\Lib\site-packages\Orange\datasets\genomic-datasets\5way\5way-best201')
    noise_features = create_noise_features(5)
    noise_data = add_noise(data, noise_features)

def create_noise_features(num):
    return [orange.FloatVariable('R' + str(i)) for i in range(1, num + 1)]

def add_noise(data, noise_features):
    npdata, np_class = data.to_numpy('a')[0], data.to_numpy('c')[0]
    length, _ = npdata.shape
    noise_data = np.random.rand(length, len(noise_features)) * 10
    np_new_data = np.concatenate((npdata, noise_data, np_class), axis=1)
    new_domain =  orange.Domain(data.domain.attributes + noise_features + [data.domain.class_var])
    return Orange.data.Table(new_domain, np_new_data)

run()