import os
import pandas as pd


    # this fun will allow us to save each model experiment result into a CSV file. 
    # If the CSV already exists, append the new result. 
    # If it does not exist, create it.


def save_experiment_results(result, results_path):

    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    new_result_df = pd.DataFrame([result])

    if os.path.exists(results_path):
        existing_results_df = pd.read_csv(results_path)
        updated_results_df = pd.concat(
            [existing_results_df, new_result_df],
            ignore_index=True
        )
    else:
        updated_results_df = new_result_df

    updated_results_df.to_csv(results_path, index=False)

    return updated_results_df
