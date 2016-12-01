import os
import ast

analysis_results_path = 'analysis.tab'

def load_results(path):
    lines = []
    with open(path) as file:
        lines = file.readlines()[1:]        

    return [get_result_from_string(line) for line in lines]

def get_result_from_string(line):
    # TODO: Add ' to save strings in file
    values = [ast.literal_eval(value) for value in line.split('\t')]
    return values

def confusion_matrix(
        relevant_variables, 
        noisy_variables, 
        scores,
        threshold
    ):

    all_variables = relevant_variables + noisy_variables
    found_variables = [all_variables[idx] for idx, var in enumerate(scores) if var > threshold]
    discarted_variables = [all_variables[idx] for idx, var in enumerate(scores) if var <= threshold]
    
    true_positives = len([variable for variable in found_variables if variable in relevant_variables])
    true_negatives = len([variable for variable in discarted_variables if variable in noisy_variables])

    false_positives = len([variable for variable in found_variables if variable in noisy_variables])
    false_negatives = len([variable for variable in discarted_variables if variable in relevant_variables])
    
    return [true_positives, true_negatives, false_positives, false_negatives]

def run_analysis():
    results = load_results('./results.tab')
    analysis = [ [file, noise, elapsed, len(relevant), len(noisy),
            confusion_matrix(relevant, noisy, r_scores, 0.0), 
            confusion_matrix(relevant, noisy, k_scores, 0.7)] 
        for [file, noise, elapsed, relevant, noisy, r_scores, k_scores] in results
        ]

    save_to_file(analysis)

def save_to_file(analysis):
    text_results = ['\t'.join([str(val) for val in data]) + '\n' for data in analysis]
    if os.path.exists(analysis_results_path):
        os.remove(analysis_results_path)

    with open(analysis_results_path, 'a') as file:
        file.writelines(text_results)

if __name__ == '__main__':
    run_analysis()