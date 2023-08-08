import pytorch_lightning as pl
import argparse
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
# import model and lightning moule
from models import TP_net_module as TPnet
from models import pde_net_module as PDEnet
# import dataloader
from models import dataset

# Other params
PLOT = True
POINT_WISE = False
ANIMATE = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TP-net params")
    parser.add_argument('--net', '-n', dest='net_name', default="TP_net", help='specify net [TP_net; PDE_net]')
    parser.add_argument('--equation', '-eq', dest='eq', default="example_256", help='specify equation data')
    parser.add_argument('--epochs', '-ep', dest='e', type=int, default=10, help='specify number of epochs')
    parser.add_argument('--neighbors', '-ns', dest='neighbors', type=int, default=25, help='specify number of neighbors')
    parser.add_argument('--gridsize', '-g', dest='grid', type=int, default=64, help='specify N points in grid NxN')
    parser.add_argument('--sample', '-s', dest='sample', type=bool, default=False, help='Sample from data for irregular grid')
    parser.add_argument('--forecast', '-f', dest='forecast', type=int, default=20, help='Forecast steps, Delta t blocks')
    parser.add_argument('--order', '-o', dest='order', type=int, default=2, help='Order of Taylor Polynomial')
    parser.add_argument('--dt', '-dt', dest='dt', type=float, default=0.1, help='Timestep dt for data')
    parser.add_argument('--test_forecast', '-tf', dest='tf', type=int, default=150, help='Test forecasting steps')
    parser.add_argument('--domain-bound', '-db', dest='domain_bound', type=int, default=32, help='Domain bound')
    args = parser.parse_args()
    EPOCHS = args.e
    EQUATION = args.eq
    NEIGHBORS = args.neighbors
    GRID_N = args.grid
    FORECAST = args.forecast
    TP_ORDER = args.order
    DOMAIN_BOUND = args.domain_bound
    DT = args.dt
    DX = DOMAIN_BOUND / GRID_N
    TEST_FORECAST = args.tf
    if args.sample == True:
        RANDOM_SAMPLE = GRID_N * GRID_N
        print(f"Random Sample {RANDOM_SAMPLE} points from data")
    else:
        RANDOM_SAMPLE = 0

if args.net_name == 'TP_net':
    ours = True
    BATCHES = 9
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-7
else:
    ours = False
    BATCHES = 2
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-6
    PDENET_KERNEL = 5
    CONSTRAINT = 'moment'

# dataset
PDE_data = dataset.pdeDataset(npy_file=EQUATION,
                      root_directory="data/" + EQUATION,
                      forecast=FORECAST,
                      DT=DT,
                      train=True,
                      k=NEIGHBORS,
                      bound=DOMAIN_BOUND,
                      pad=True,
                      random_sample=RANDOM_SAMPLE,
                      TP_order=TP_ORDER,
                      animate=ANIMATE
                      )
PDE_data_test = dataset.pdeDataset(npy_file=EQUATION,
                      root_directory="data/" + EQUATION,
                      forecast=TEST_FORECAST,
                      DT=DT,
                      train=False,
                      k=NEIGHBORS,
                      bound=DOMAIN_BOUND,
                      pad=True,
                      random_sample=RANDOM_SAMPLE,
                      TP_order=TP_ORDER,
                      animate=ANIMATE
                      )

PDE_train = DataLoader(PDE_data, batch_size=BATCHES, shuffle=True, num_workers=8)
PDE_val   = DataLoader(PDE_data, batch_size=1, shuffle=False, num_workers=8)
PDE_test  = DataLoader(PDE_data_test, batch_size=1, shuffle=False, num_workers=8)
# Logger
tb_logger  = pl_loggers.TensorBoardLogger(save_dir="outputs/" + args.net_name + "/tb_logs/" + EQUATION)
csv_logger = pl_loggers.CSVLogger(save_dir="outputs/" + args.net_name + "/csv_logs/" + EQUATION)
# Trainer
trainer = pl.Trainer(logger=[tb_logger, csv_logger],
                     max_epochs=EPOCHS,
                     accelerator="gpu",
                     devices=1,
                     log_every_n_steps=1,
                     limit_val_batches=1,
                     limit_test_batches=1,
                     val_check_interval=0.05
                     )
# net
if ours:
    # net
    net = TPnet.TPNet(N=GRID_N,
                    num_neighbors=NEIGHBORS,
                    TP_order=TP_ORDER,
                    point_wise=POINT_WISE,
                    plot_deriv=PLOT)
    # Lightning model
    model = TPnet.TPNetModule(net=net,
                    learning_rate=LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY,
                    plot=PLOT,
                    point_wise=POINT_WISE,
                    N=GRID_N,
                    equation=EQUATION)
else:
    # net
    net = PDEnet.PDENet(input_dim=1,
                     kernel_size=PDENET_KERNEL,
                     max_order=TP_ORDER,
                     constraint=CONSTRAINT,
                     hidden_layers=1,
                     scheme='upwind',
                     dt=DT,
                     dx=DX
                     )
    # Lightning model
    model = PDEnet.PDENetModule(net=net,
                               learning_rate=LEARNING_RATE,
                               weight_decay=WEIGHT_DECAY,
                               plot=PLOT,
                               point_wise=POINT_WISE,
                               N=GRID_N,
                               equation=EQUATION)

# Log Hyperparms
hyperparameters = dict(batches=BATCHES,
                       forecast=FORECAST,
                       epochs=EPOCHS,
                       neighbors=NEIGHBORS,
                       learning_rate=LEARNING_RATE,
                       weight_decay=WEIGHT_DECAY,
                       point_wise=POINT_WISE,
                       grid_n=GRID_N,
                       tp_order=TP_ORDER,
                       equation=EQUATION,
                       test_forecast=TEST_FORECAST)

trainer.logger.log_hyperparams(hyperparameters)
# train model
trainer.fit(model, PDE_train, PDE_val)
# test model
trainer.test(model, PDE_test)
