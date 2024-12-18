import os
import glob
import torch
from cace.tasks import LightningData, LightningTrainingTask

root = "data/spice-dipep-dipolar.h5"

LR = True
logs_name = "cace_dipep_c4_dipolar"
if LR:
    logs_name += "_lr"
else:
    logs_name += "_sr"
cutoff = 4.0
batch_size = 4
from dipeptides.data import SpiceData
data = SpiceData(root,cutoff,batch_size=batch_size,in_memory=True,valid_p=0.1,test_p=0)

from cace.representations import Cace
from cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered
from cace.modules import PolynomialCutoff

#Model
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

representation = Cace(
    zs=[1,6,7,8],
    n_atom_basis=4,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=12,
    max_l=4,
    max_nu=3,
    num_message_passing=1,
    type_message_passing=["M", "Ar", "Bchi"],
    args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
    timeit=False
)

import cace
from cace.models import NeuralNetworkPotential
from cace.modules import Atomwise, Forces

atomwise = Atomwise(n_layers=3,
                    output_key="pred_energy",
                    n_hidden=[32,16],
                    n_out=1,
                    use_batchnorm=False,
                    add_linear_nn=True)

forces = Forces(energy_key="pred_energy",
                forces_key="pred_force")

model = NeuralNetworkPotential(
    input_modules=None,
    representation=representation,
    output_modules=[atomwise,forces]
)

if LR:
    q = cace.modules.Atomwise(
        n_layers=3,
        n_hidden=[24,12],
        n_out=1,
        per_atom_output_key='q',
        output_key = 'tot_q',
        residual=False,
        add_linear_nn=True,
        bias=False)
    
    ep = cace.modules.EwaldPotential(dl=3,
                        sigma=1.5,
                        feature_key='q',
                        output_key='ewald_potential',
                        remove_self_interaction=False,
                       aggregation_mode='sum')
    
    forces_lr = cace.modules.Forces(energy_key='ewald_potential',
                                        forces_key='ewald_forces')
    
    cace_nnp_lr = NeuralNetworkPotential(
        input_modules=None,
        representation=representation,
        output_modules=[q, ep, forces_lr]
    )

    pot1 = {'pred_energy': 'pred_energy', 
            'pred_force': 'pred_force',
            'weight': 1,
           }
    
    pot2 = {'pred_energy': 'ewald_potential', 
            'pred_force': 'ewald_forces',
            'weight': 1,
           }
    
    model = cace.models.CombinePotential([model, cace_nnp_lr], [pot1,pot2])

#Losses
from cace.tasks import GetLoss
e_loss = GetLoss(
    target_name="energy",
    predict_name='pred_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1,
)
f_loss = GetLoss(
    target_name="force",
    predict_name='pred_force',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000,
)
losses = [e_loss,f_loss]

#Metrics
from cace.tools import Metrics
e_metric = Metrics(
            target_name="energy",
            predict_name='pred_energy',
            name='e',
            metric_keys=["rmse"],
            per_atom=True,
        )
f_metric = Metrics(
            target_name="force",
            predict_name='pred_force',
            metric_keys=["rmse"],
            name='f',
        )
metrics = [e_metric,f_metric]

#Init lazy layers
for batch in data.train_dataloader():
    exdatabatch = batch
    break
model(exdatabatch)

#Check for checkpoint and restart if found:
chkpt = None
dev_run = False
if os.path.isdir(f"lightning_logs/{logs_name}"):
    latest_version = None
    num = 0
    while os.path.isdir(f"lightning_logs/{logs_name}/version_{num}"):
        latest_version = f"lightning_logs/{logs_name}/version_{num}"
        num += 1
    if latest_version:
        chkpt = glob.glob(f"{latest_version}/checkpoints/*.ckpt")[0]
if chkpt:
    print("Checkpoint found!",chkpt)
    print("Restarting...")
    dev_run = False

progress_bar = True
torch.set_float32_matmul_precision('medium')
task = LightningTrainingTask(model,losses=losses,metrics=metrics,
                             logs_directory="lightning_logs",name=logs_name,
                             scheduler_args={'mode': 'min', 'factor': 0.8, 'patience': 10},
                             optimizer_args={'lr': 0.01},
                            )
task.fit(data,dev_run=dev_run,max_epochs=500,chkpt=chkpt,progress_bar=progress_bar)