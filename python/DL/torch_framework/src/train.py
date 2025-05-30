import os, sys
from utils.utils import *
from utils.model_utils import *
from dataset.patDataset import *
from torch.utils.data import DataLoader


def train(config):
    
    recorder = config["recorder"]
    
    train_dataset = patDataset(
        data_dir = config["data_root_dir"],
        is_train = True,
        regex_pattern = [f"({'|'.join(config['train_data_ID'])}){x}" for x in config["train_regex_pattern"]],
        process_fun   = config["method_function"].train_dataset_step,
        args = config
    )
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = config["train_batch_size"],
        shuffle = True
    )
    
    val_dataset = patDataset(
        data_dir = config["data_root_dir"],
        is_train = True,
        regex_pattern = [f"({'|'.join(config['val_data_ID'])}){x}" for x in config["val_regex_pattern"]],
        process_fun   = config["method_function"].val_dataset_step,
        args = config
    )
    val_dataloader = DataLoader(
        dataset = val_dataset,
        batch_size = config["val_batch_size"],
        shuffle = False
    )
    
    len_train_dataset    = len(train_dataset)
    len_train_dataloader = len(train_dataloader)
    len_val_dataset      = len(val_dataset)
    len_val_dataloader   = len(val_dataloader)
    
    
    recorder.message(
        f"""
        \rtask name: {config['task_name']}
        \ruse_device: {config['device']}
        \rModel: {config['model']}
        \rOptimizer: {config['optimizer']}
        \rLearning rate scheduler: {config['lr_scheduler']}
        \rTrain batch size: {config['train_batch_size']}
        \rVal batch size: {config['val_batch_size']}
        \rLearning rate: {config['learning_rate']}
        \rNumber of training data: {len_train_dataset}
        \rNumber of validation data: {len_val_dataset}
        """
    )
    
    models     = create_model(config)
    bestmodels = None
    optimizers = create_optimizer(config, models)
    schedulers = create_scheduler(config, optimizers)
    
    train_loss    = {}
    val_loss      = {}
    val_indicator = {}
    
    if config["supervision_strategy"] == "max": monitoring_indicators = sys.float_info.min
    if config["supervision_strategy"] == "min": monitoring_indicators = sys.float_info.max
    
    start_epoch = -1
    if config["is_continue"] == True:
        models, optimizers, start_epoch, _, info = load_model(
            models = models,
            optimizers = optimizers,
            config = config
        )
        recorder.message(info)
    else: recorder.message("Do not load the model, it will be retrained")
    
    for epoch_idx in range(start_epoch + 1, config["epochs"]):
        epoch_train_loss    = {}
        epoch_val_loss      = {}
        epoch_val_indicator = {}
        
        # train
        for step, data in enumerate(train_dataloader):
            train_step_loss = config["method_function"].train_step(models, optimizers, data, step, epoch_idx, config)
            
            epoch_train_loss = dictionary_addition(
                A = epoch_train_loss,
                B = dictionary_division(train_step_loss, len_train_dataloader)
            )
            recorder.message(f"epoch {epoch_idx} / {config['epochs'] - 1}, step {step} / {len_train_dataloader - 1}, {dict2str(train_step_loss)}", end = "\r")
        train_loss = dict_append(train_loss, epoch_train_loss)
        recorder.message(f"epoch {epoch_idx}, {dict2str(train_step_loss)}")
        
        # validation
        for step, data in enumerate(val_dataloader):
            val_step_loss, val_step_indicator = config["method_function"].val_step(models, optimizers, data, step, epoch_idx, config)
            epoch_val_loss = dictionary_addition(
                A = epoch_val_loss,
                B = dictionary_division(val_step_loss, len_val_dataloader)
            )
            epoch_val_indicator = dictionary_addition(
                A = epoch_val_indicator,
                B = dictionary_division(val_step_indicator, len_val_dataloader)
            )
            recorder.message(f"epoch {epoch_idx}/{config['epochs'] - 1}, step {step}/{len_val_dataloader - 1}, {dict2str(val_step_loss)}, {dict2str(val_step_indicator)}", end = "\r", stage = "val", state = "val")
        val_loss = dict_append(val_loss, epoch_val_loss)
        val_indicator = dict_append(val_indicator, epoch_val_indicator)
        recorder.message(f"epoch {epoch_idx}, {dict2str(val_step_loss)}, {dict2str(val_step_indicator)}", stage = "val", state = "val")
        
        schedulers_step(schedulers=schedulers)
# dictData, x_labels, y_labels, subtitle, title, shape,  save_path, save_name
        recorder.draw(
            dictData = train_loss, 
            title = "train loss", 
            save_path = os.path.join(config["root_dir"], "log", config["task_name"]),
            save_name = "train_loss",
            x_labels = [], y_labels = [], subtitle = [], shape = None
        )
        recorder.draw(
            dictData = val_loss,
            title = "val loss", 
            save_path = os.path.join(config["root_dir"], "log", config["task_name"]),
            save_name = "val_loss",
            x_labels = [], y_labels = [], subtitle = [], shape = None
        )
        recorder.draw(
            dictData = val_indicator, 
            title = "val indicator", 
            save_path = os.path.join(config["root_dir"], "log", config["task_name"]),
            save_name = "val_indicator",
            x_labels = [], y_labels = [], subtitle = [], shape = None
        )
        
        if check(epoch_val_indicator[config["monitoring_indicators"]], monitoring_indicators, config["supervision_strategy"]):
            monitoring_indicators = epoch_val_indicator[config["monitoring_indicators"]]
            best_models = models.copy()
            info = save_model(
                epoch = epoch_idx,
                indicator = epoch_val_indicator,
                models = models,
                optimizers = optimizers,
                config = config
            )
            recorder.message(info)
            
    recorder.message(f"Training completed!")
    save_model(
        models = best_models,
        save_name = "best_model.pth",
        config = config
    )
         
        
        
                
        
        
        
        
        
        
        
        
    