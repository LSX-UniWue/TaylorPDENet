import pytorch_lightning as pl
import optuna
import argparse
import logging
import sys
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
# import model and lightning moule
from models import pde_net_module as PDEnet
# import dataloader
from models import dataset

#Hyperparameters
EQUATION = "example2_64"
BATCHES = 4
FORECAST = 20
EPOCHS = 10
GRID_N = 64
NEIGHBORS = 16
TP_ORDER = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
POINT_WISE = False
PLOT = False
ANIMATE = False
RANDOM_SAMPLE = 0
DOMAIN_BOUND = 32 # can be found in pde-generator Extend of Grid
DT = 0.1
DX = DOMAIN_BOUND / GRID_N

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

def objective(trial: optuna.trial.Trial) -> float:
    BATCHES = trial.suggest_int("batch_seize", 1, 16)
    WD_BOOL = trial.suggest_categorical("wd_bool", [1, 0])
    CONSTRAINT = trial.suggest_categorical("constraint", ['FROZEN', 'moment'])
    KERNEL_S = trial.suggest_categorical("kernel_size", [3, 5, 7])
    if WD_BOOL == 1:
        WEIGHT_DECAY = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    else:
        WEIGHT_DECAY = 0.0
    LEARNING_RATE = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    #POINT_WISE = trial.suggest_categorical("point_wise", [True, False]) # maybe

    # net
    net = PDEnet.PDENet(input_dim=1,
                     kernel_size=KERNEL_S,
                     max_order=2,
                     constraint=CONSTRAINT,
                     hidden_layers=4,
                     scheme='upwind',
                     dt=DT,
                     dx=DX)
    # Lightning model
    model = PDEnet.PDENetModule(net=net,
                               learning_rate=LEARNING_RATE,
                               weight_decay=WEIGHT_DECAY,
                               plot=PLOT,
                               point_wise=POINT_WISE,
                               N=GRID_N)
    
    PDE_train = DataLoader(PDE_data, batch_size=BATCHES, shuffle=True, num_workers=8)
    PDE_val = DataLoader(PDE_data, batch_size=1, shuffle=False, num_workers=8)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="outputs/study_PDE/tb_logs_" + EQUATION)
    csv_logger = pl_loggers.CSVLogger(save_dir="outputs/study_PDE/csv_logs_"+EQUATION)
    
    trainer = pl.Trainer(logger=[tb_logger, csv_logger],
                        max_epochs=EPOCHS,
                        accelerator="gpu",
                        devices=1,
                        log_every_n_steps=1,
                        #limit_train_batches=1, # del when study
                        limit_val_batches=1,
                        val_check_interval=0.25
                        )

    hyperparameters = dict(batches=BATCHES,
                           forcast=FORECAST,
                           epochs=EPOCHS,
                           neighbors=NEIGHBORS,
                           learning_rate=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY,
                           point_wise=POINT_WISE,
                           grid_n=GRID_N,
                           tp_order=TP_ORDER,
                           kernel_size=KERNEL_S,
                           constraint=CONSTRAINT
                           )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, PDE_train, PDE_val)

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDE-net params")
    parser.add_argument(
        '--equation',
        dest='eq',
        action='store',
        default="example2_64",
        help='specify equation data'
    )
    args = parser.parse_args()

    EQUATION = args.eq
    
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "PDE-net-Hyperparameter-search"
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
