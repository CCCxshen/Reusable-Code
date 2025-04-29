import lightning as L
from torch.nn import functional as F
from utils.loss import *
from utils.about_model import *

class lightning_model(L.LightningModule):

    def __init__(self, model, opt):
        self.opt = opt
        self.model = model
        
        self.indicator = None

        if opt["about_lt_model"]["loss"] == "MSE":
            self.loss_fn = MSELoss()
        elif opt["about_lt_model"]["loss"] == "L1":
            self.loss_fn = L1Loss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = obtain_optim(self.opt, self.parameters())
        scheduler = obtain_scheduler(self.opt, optimizer)
        return [optimizer], [scheduler]
            