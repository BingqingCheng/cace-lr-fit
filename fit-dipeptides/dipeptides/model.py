import os
import glob
import torch
from cace.tasks import LightningData, LightningTrainingTask
from cace.representations import Cace
from cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered
from cace.modules import PolynomialCutoff

def make_cace_lr(cutoff=4.0,lr=True):
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
    
    if lr:
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

    return model