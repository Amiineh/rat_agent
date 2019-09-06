import json


def run(output_path, agent):
    if agent == 'priint':
        import agents.priint.experiments as experiments

    experiments.generate_experiments(output_path)


