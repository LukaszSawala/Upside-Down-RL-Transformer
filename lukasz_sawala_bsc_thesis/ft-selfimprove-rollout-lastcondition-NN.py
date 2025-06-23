import numpy as np
import gymnasium as gym
import gymnasium_robotics
import torch
from scipy.stats import sem
from dataset_generation import generate_dataset
from finetuningExtraDataAntmaze import grid_search_experiment_from_rollout
from transfer_eval_main import antmaze_evaluate, transfer_eval_main
from model_evaluation_ALL import plot_all_models_rewards
from dataset_generation import INITIAL_ANTMAZE_NN_PATH


NUMBER_OF_ITERATIONS = 6  # Set carefully! Every iteration will take a long time (~2-3h hours) to complete.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # --- Parameters ---
    d_h = 1000.0
    d_r_options = [i * 50 for i in range(19)]
    num_episodes_per_dr = 50 
    # --- This is a standard setting yielding 1000*19*50 = ~1m transitions in total.
    # --- Final dataset size will differ - 90% of low reward episodes are removed to increase stability.

    # --- Hyperparameters for grid search ---
    batch_sizes_param = [16]
    learning_rates_param = [5e-5]
    epochs_list_param = [100]

    #model_to_use = "ANTMAZE_BERT_MLP"
    model_to_use = "ANTMAZE_NN" 

    # --- Results storage ---
    results = {}
    for i in range(NUMBER_OF_ITERATIONS+1):
        results[f"NeuralNet{i}"] = {
            "avg_rewards": [],
            "sem": [],
            "success_rates": []
        }

    gym.register_envs(gymnasium_robotics)
    env = gym.make("AntMaze_MediumDense-v5")


    name = f"NeuralNet0"
    print(f"Evaluating base model: {name}")
    args = {
        "model_type": model_to_use,
        "model_path": INITIAL_ANTMAZE_NN_PATH,
        "d_r_array_length": len(d_r_options),
        "episodes": 10,
        "return_without_plotting": True
    }
    avg, se, success_rates = transfer_eval_main(args)
    results[name]["avg_rewards"] = avg
    results[name]["sem"] = se
    results[name]["success_rates"] = success_rates

    # --- Start the iterative process ---
    start_from_condition4 = True  # condition 4 uses the model trained on all antmaze datasets
    for i in range(NUMBER_OF_ITERATIONS):
        print("=" * 60)
        print(f"Iteration {i+1}/{NUMBER_OF_ITERATIONS}")
        # --- 1. Generate dataset ---
        generate_dataset(d_h=d_h, d_r_options=d_r_options,
                         num_episodes_per_dr=num_episodes_per_dr,
                         start_from_condition4=start_from_condition4, retain_best_previous_data=True)
        # This will update the dataset used in the next step.
        
        # --- 2. Finetune by Grid search  ---
        model = grid_search_experiment_from_rollout(batch_sizes_param=batch_sizes_param,
                           learning_rates_param=learning_rates_param,
                           epochs_list_param=epochs_list_param,
                           model_to_use=model_to_use,
                           start_from_condition4=start_from_condition4)

        # --- 3. Evaluate the new model ---
        name = f"NeuralNet{i+1}"
        print(f"Evaluating model: {name}")
        for d_r in d_r_options:
            print("=" * 50)
            print(f"Evaluating d_r: {d_r}")
            returns, distances = antmaze_evaluate(env, model, episodes=10, d_r=d_r,
                                                d_h=d_h, state_dim=27, use_goal=True)
            avg = np.mean(returns)
            se = sem(returns)
            results[name]["avg_rewards"].append(avg)
            results[name]["sem"].append(se)
            results[name]["success_rates"].append(np.mean([d < 1 for d in distances]))
        start_from_condition4 = False  # After the first iteration, we do not need to start from condition 4 anymore.

    env.close()
    
    print("\n" + "=" * 60)
    print("Final Success Rates per Model:")
    for model_name, data in results.items():
        print(f"{model_name}: {np.mean(data['success_rates'])*100:.2f}%")

    # plot only each 3rd model
    results_to_plot = {model_name: data for i, (model_name, data) in enumerate(results.items()) if i != 3}
    # Final multi-model plot
    save_path = f"condition5-{model_to_use}.png"
    plot_all_models_rewards(results_to_plot, d_r_options, save_path=save_path)