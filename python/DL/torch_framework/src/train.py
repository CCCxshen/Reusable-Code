import os, sys
from datetime import datetime
from utils.utils import *
from utils.IO_utils import *
from utils.ddp_utils import *
from utils.model_utils import *
from dataset.patDataset import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def train(config):
    
    recorder = config["recorder"]
    
    train_dataset = patDataset(
        data_dir = config["train_data_root_dir"],
        is_train = True,
        regex_pattern = [f"({'|'.join(config['train_data_ID'])}){x}" for x in config["train_regex_pattern"]],
        process_fun   = config["method_function"].train_dataset_step,
        args = config
    )
    
    if config["DDP"] == True:
        train_sampler = DistributedSampler(
            dataset = train_dataset,
            num_replicas = config["world_size"],
            rank = config["local_rank"],
            shuffle = True
        )
    
        train_dataloader = DataLoader(
            dataset = train_dataset,
            batch_size = config["train_batch_size"],
            sampler = train_sampler,
        )
    else:
        train_dataloader = DataLoader(
            dataset = train_dataset,
            batch_size = config["train_batch_size"],
            shuffle = True
        )
    
    
    val_dataset = patDataset(
        data_dir = config["val_data_root_dir"],
        is_train = False,
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
        \rmethod_name: {config['method_name']}
        \ruse_device: {config['device']}
        \rModel: {config['model']}
        \rOptimizer: {config['optimizer']}
        \rLearning rate scheduler: {config['lr_scheduler']}
        \rTrain batch size: {config['train_batch_size']}
        \rVal batch size: {config['val_batch_size']}
        \rLearning rate: {config['learning_rate']}
        \rNumber of training data: {len_train_dataset}
        \rNumber of validation data: {len_val_dataset}
        \rStrict loading model: {config['strict_loading_model_weight']}
        """,
        ignore_width = True
    )
    
    models     = create_model(config)
    bestmodels = None
    optimizers = create_optimizer(config, models)
    schedulers = create_scheduler(config, optimizers)
    if 'indicator_name' in config:
        config['indicator'] = create_indicator(config)
    if 'loss' in config:
        config['loss'] = create_loss(config)
    
    train_loss    = load_dict_from_file(os.path.join(config["etc_path"], "train_loss.npz"))
    val_loss      = load_dict_from_file(os.path.join(config["etc_path"], "val_loss.npz"))
    val_indicator = load_dict_from_file(os.path.join(config["etc_path"], "val_indicator.npz"))
    
    if config["supervision_strategy"] == "max": monitoring_indicators = sys.float_info.min
    if config["supervision_strategy"] == "min": monitoring_indicators = sys.float_info.max
    monitoring_indicators_info = load_dict_from_file(os.path.join(config["etc_path"], "monitoring_indicators_info.npz"))
    if (monitoring_indicators_info != {}) and (monitoring_indicators_info['monitoring_indicators_name'] == config["monitoring_indicators"] and monitoring_indicators_info['supervision_strategy'] == config["supervision_strategy"]):
        monitoring_indicators = monitoring_indicators_info['monitoring_indicators']
    
    start_epoch = -1
    if config["is_continue"] == True:
        models, optimizers, start_epoch, _, info = load_model(
            models = models,
            optimizers = optimizers,
            config = config
        )
        recorder.message(info, ignore_width = True)
    else: recorder.message("Do not load the model, it will be retrained", ignore_width = True)
    
    train_start_time = datetime.now()
    for epoch_idx in range(start_epoch + 1, config["epochs"]):
        epoch_start_time = datetime.now()
        
        epoch_train_loss    = {}
        epoch_val_loss      = {}
        epoch_val_indicator = {}
        
        # train
        recorder.message(f"epoch {epoch_idx}/{config['epochs'] - 1}, step {0}/{len_train_dataloader - 1}", end = "\r", write_into = "never")
        for step, data in enumerate(train_dataloader):
            
            train_step_loss = config["method_function"].train_step(models, optimizers, data, step, epoch_idx, config)
            recorder.message(f"epoch {epoch_idx}/{config['epochs'] - 1}, step {step}/{len_train_dataloader - 1}, {dict2str(train_step_loss)}", end = "\r")
            
            epoch_train_loss = dictionary_addition(
                A = epoch_train_loss,
                B = dictionary_division(train_step_loss.copy(), len_train_dataloader)
            )
        
        train_loss = dict_append(train_loss, epoch_train_loss)
        recorder.message(f"epoch {epoch_idx}, {dict2str(epoch_train_loss)}", ignore_width = True)
        
        # validation
        if (config["DDP"] == False) or (config["DDP"] == True and int(os.environ["RANK"]) == 0):
            recorder.message(f"epoch {epoch_idx}/{config['epochs'] - 1}, step {0}/{len_val_dataloader - 1}", end = "\r", stage = "val", write_into = "never")
            for step, data in enumerate(val_dataloader):
                val_step_loss, val_step_indicator = config["method_function"].val_step(models, data, step, epoch_idx, config)
                recorder.message(f"epoch {epoch_idx}/{config['epochs'] - 1}, step {step}/{len_val_dataloader - 1}, {dict2str(val_step_loss)}, {dict2str(val_step_indicator)}", end = "\r", stage = "val", write_into = "val")
                epoch_val_loss = dictionary_addition(
                    A = epoch_val_loss,
                    B = dictionary_division(val_step_loss, len_val_dataloader)
                )
                epoch_val_indicator = dictionary_addition(
                    A = epoch_val_indicator,
                    B = dictionary_division(val_step_indicator, len_val_dataloader)
                )
                
            val_loss = dict_append(val_loss, epoch_val_loss)
            val_indicator = dict_append(val_indicator, epoch_val_indicator)
            recorder.message(f"epoch {epoch_idx}, {dict2str(epoch_val_loss)}, {dict2str(epoch_val_indicator)}", stage = "val", write_into = "val", ignore_width = True)
        
        if schedulers != None: schedulers_step(schedulers=schedulers)
        
        epoch_end_time = datetime.now()
        train_spent_time, epoch_spent_time, remain_spent_time = time_utils(train_start_time, epoch_start_time, epoch_end_time, config["epochs"] - epoch_idx - 1)
        recorder.message(f"epoch {epoch_idx} time is {epoch_spent_time}, total time is {train_spent_time}, remaining time is {remain_spent_time}", ignore_width = True)

        recorder.draw(
            dictData = train_loss, 
            title = "train loss", 
            save_path = config["log_path"],
            save_name = "train_loss",
            x_labels = [], y_labels = [], subtitle = [], shape = None
        )
        recorder.draw(
            dictData = val_loss,
            title = "val loss", 
            save_path = config["log_path"],
            save_name = "val_loss",
            x_labels = [], y_labels = [], subtitle = [], shape = None
        )
        recorder.draw(
            dictData = val_indicator, 
            title = "val indicator", 
            save_path = config["log_path"],
            save_name = "val_indicator",
            x_labels = [], y_labels = [], subtitle = [], shape = None
        )
        
        if ((config["DDP"] == False) or (config["DDP"] == True and int(os.environ["RANK"]) == 0)):
            # 保存最新模型
            save_model(
                save_name = "last_model.pth",
                epoch = epoch_idx,
                indicator = epoch_val_indicator,
                models = models,
                optimizers = optimizers,
                config = config,
            )
            
            better_model = False
            # 保存最好的模型
            if check(epoch_val_indicator[config["monitoring_indicators"]], monitoring_indicators, config["supervision_strategy"]):
                save_model(
                    models = models,
                    save_name = "best_model.pth",
                    config = config,
                )
                monitoring_indicators = epoch_val_indicator[config["monitoring_indicators"]]
                better_model = True
            
            # 保存每一个模型，或者保存更好的模型
            if (config["save_every_model"] == True or better_model == True):
                info = save_model(
                    epoch = epoch_idx,
                    indicator = epoch_val_indicator,
                    models = models,
                    optimizers = optimizers,
                    config = config,
                )
                recorder.message(info, ignore_width = True)
        
        save_dict_to_file(train_loss, os.path.join(config["etc_path"], "train_loss"))
        save_dict_to_file(val_loss, os.path.join(config["etc_path"], "val_loss"))
        save_dict_to_file(val_indicator, os.path.join(config["etc_path"], "val_indicator"))
        save_dict_to_file(
            {
                "monitoring_indicators_name": config["monitoring_indicators"], 
                "supervision_strategy": config["supervision_strategy"],
                "monitoring_indicators": monitoring_indicators
            },
            os.path.join(config["etc_path"], "monitoring_indicators_info")
        )
        
    recorder.message(f"Training completed!", ignore_width = True)
    
    if config["DDP"] == True:
        cleanup_ddp()    