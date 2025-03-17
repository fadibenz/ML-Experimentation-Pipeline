from utils.utils import set_seed
from scripts.objective import objective
import optuna
from scripts.make_predict import predict

if __name__ == '__main__':
    set_seed()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10, timeout=3600)

    print('Best trial:')
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


    pass