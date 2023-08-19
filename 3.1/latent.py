# Get latent vectors for mp_20 dataset

import os

os.environ["PROJECT_ROOT"] = "/jmain02/home/J2AD015/axf03/hxf45-axf03/src/cdvae"
os.environ["HYDRA_JOBS"] = "/jmain02/home/J2AD015/axf03/hxf45-axf03/src/cdvae"
os.environ["WABDB_DIR"] = "/jmain02/home/J2AD015/axf03/hxf45-axf03/src/cdvae"

from hydra.experimental import compose
from hydra import initialize_config_dir
import hydra
from pathlib import Path
import numpy as np
import torch
import cdvae
import pandas as pd

model_path = Path("/jmain02/home/J2AD015/axf03/hxf45-axf03/src/cdvae/singlerun/2023-05-18/mp_20")
with initialize_config_dir(str(model_path)):
    cfg = compose(config_name='hparams')

    # load model
    model = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )
    ckpts = list(model_path.glob('*.ckpt'))
    if len(ckpts) > 0:
        ckpt_epochs = np.array(
            [int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
        ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
    model = model.load_from_checkpoint(ckpt)
    model.lattice_scaler = torch.load(model_path / 'lattice_scaler.pt')
    model.scaler = torch.load(model_path / 'prop_scaler.pt')

datamodule = hydra.utils.instantiate(
    cfg.data.datamodule, _recursive_=False, scaler_path=model_path
)
datamodule.setup()
test_loader = datamodule.test_dataloader()[0]    
train_loader = datamodule.train_dataloader()  
val_loader = datamodule.val_dataloader()[0]

test_batches = list(test_loader)
train_batches = list(train_loader)
val_batches = list(val_loader)

#combine sets of batches
batches = (test_batches, train_batches, val_batches)
mu_t , log_t, z_t = torch.tensor([]), torch.tensor([]), torch.tensor([])
for item in batches:
    for batch in item:
        mu, log, z = model.encode(batch)
        mu_t = torch.cat([mu_t , mu])
        log_t = torch.cat([log_t , log])
        z_t = torch.cat([z_t , z])

mu, log, z = mu_t.detach().numpy(), log_t.detach().numpy(), z_t.detach().numpy()
pd.DataFrame(mu).to_csv('mu.csv',index=False)
pd.DataFrame(log).to_csv('log_var.csv',index=False)
pd.DataFrame(z).to_csv(str('z.csv',index=False)
