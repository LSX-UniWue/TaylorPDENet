import pytorch_lightning as pl
import optuna
import argparse
import logging
import sys
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
# import model and lightning moule
from models import TP_net_module as TP
# import dataloader
from models import dataset

#Hyperparameters
FORECAST = 20
EPOCHS = 10
GRID_N = 64
TP_ORDER = 2
POINT_WISE = False
PLOT = False
ANIMATE = False
RANDOM_SAMPLE = GRID_N*GRID_N
DOMAIN_BOUND = 32 # can be found in pde-generator Extend of Grid
DT = 0.1

def objective(trial: optuna.trial.Trial) -> float:
    BATCHES = trial.suggest_int("batch_seize", 1, 16)
    NEIGHBORS = trial.suggest_int("neighbors", 10, 20)
    WEIGHT_DECAY = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    LEARNING_RATE = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    #POINT_WISE = trial.suggest_categorical("point_wise", [True, False]) # maybe

    # dataset
    PDE_data = dataset.pdeDataset(npy_file=EQUATION,
                        root_directory="data/" + EQUATION,
                        forecast=FORECAST,
                        DT=DT,
                        train=True,
                        k=NEIGHBORS,
                        random_sample=RANDOM_SAMPLE,
                        TP_order=TP_ORDER,
                        animate=ANIMATE
                        )

    net = TP.TPNet(N=GRID_N,
                       num_neighbors=NEIGHBORS,
                       TP_order=TP_ORDER,
                       point_wise=POINT_WISE,
                       plot_deriv=PLOT)
    
    model = TP.TPNetModule(net,
                        learning_rate=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY,
                        plot=PLOT,
                        point_wise=POINT_WISE,
                        N=GRID_N)
    
    PDE_train = DataLoader(PDE_data, batch_size=BATCHES, shuffle=True, num_workers=8)
    PDE_val = DataLoader(PDE_data, batch_size=1, shuffle=False, num_workers=8)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="outputs/study_TP/tb_logs_" + EQUATION)
    csv_logger = pl_loggers.CSVLogger(save_dir="outputs/study_TP/csv_logs_"+EQUATION)
    
    trainer = pl.Trainer(logger=[tb_logger, csv_logger],
                        max_epochs=EPOCHS,
                        accelerator="gpu",
                        devices=1,
                        log_every_n_steps=1,
                        limit_val_batches=1,
                        val_check_interval=0.05
                        )

    hyperparameters = dict(batches=BATCHES,
                           forcast=FORECAST,
                           epochs=EPOCHS,
                           neighbors=NEIGHBORS,
                           learning_rate=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY,
                           point_wise=POINT_WISE,
                           grid_n=GRID_N,
                           tp_order=TP_ORDER
                           )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, PDE_train, PDE_val)

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TP-net params")
    parser.add_argument(
        '--equation',
        dest='eq',
        action='store',
        default="example5_256",
        help='specify equation data'
    )
    args = parser.parse_args()

    EQUATION = args.eq
    
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "TP-net-Hyperparameter-search"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        direction="minimize",
        storage=storage_name,
        study_name=study_name
        )

    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
