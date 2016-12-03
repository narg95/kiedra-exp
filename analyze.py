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

    return [true_positives, true_negatives]

def run_analysis():
    results = load_results('./results.tab')
    analysis = [ [file,
            confusion_matrix(relevant, noisy, r_scores, 0.0), 
            confusion_matrix(relevant, noisy, k_scores, 0.7)] 
        for [file, _, _, relevant, noisy, r_scores, k_scores] in results
        ]

    aggregated_result = {}
    for [file, [r_tp, r_tn], [k_tp, k_tn]] in analysis:
        problem_key = file.split('-')[0]
        key = (problem_key, r_tp, r_tn)

        r_acc, k_acc, acc = calculate_accumulated_values(aggregated_result, key, False)
        aggregated_result[key] = [problem_key, r_tp, r_tn, r_acc, k_acc, acc]

        key = (problem_key, k_tp, k_tn)
        r_acc, k_acc, acc = calculate_accumulated_values(aggregated_result, key, True)
        aggregated_result[key] = [problem_key, k_tp, k_tn, r_acc, k_acc, acc]
    
    save_to_file(aggregated_result.values())

def calculate_accumulated_values(aggregated_result, key, is_kiedra):
    if key not in aggregated_result:
        r_acc = 0 if is_kiedra else 1
        k_acc = 1 if is_kiedra else 0
        return r_acc, k_acc, 1

    [r_acc, k_acc, acc] = aggregated_result[key][3:]      
    r_acc = r_acc if is_kiedra else r_acc + 1
    k_acc = k_acc + 1 if is_kiedra else k_acc
    acc = acc + 1
    return r_acc, k_acc, acc


def save_to_file(analysis):
    text_results = ['\t'.join([str(val) for val in data]) + '\n' for data in analysis]
    if os.path.exists(analysis_results_path):
        os.remove(analysis_results_path)

    with open(analysis_results_path, 'a') as file:
        file.writelines(text_results)

if __name__ == '__main__':
    run_analysis()