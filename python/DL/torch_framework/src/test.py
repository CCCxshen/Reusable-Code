import os, sys
from utils.utils import *
from utils.IO_utils import *
from utils.ddp_utils import *
from utils.model_utils import *
from dataset.patDataset import *
from torch.utils.data import DataLoader


def test(config):
    
    recorder = config["recorder"]
    
    test_dataset = patDataset(
        data_dir = config["test_data_root_dir"],
        is_train = True,
        regex_pattern = [f"({'|'.join(config['test_data_ID'])}){x}" for x in config["test_regex_pattern"]],
        process_fun   = config["method_function"].test_dataset_step,
        args = config
    )
    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = config["test_batch_size"],
        shuffle = True
    )
    
    len_test_dataset    = len(test_dataset)
    len_test_dataloader = len(test_dataloader)
    
    recorder.message(
        f"""
        \rmethod name: {config['method_name']}
        \ruse_device: {config['device']}
        \rModel: {config['model']}
        \rTest batch size: {config['test_batch_size']}
        \rNumber of test data: {len_test_dataset}
        \rstart test
        """,
        ignore_width = True,
        stage = "test",
        write_into = "test"
    )
    
    models = create_model(config)
    if 'indicator_name' in config:
        config['indicator'] = create_indicator(config)
    
    models, _, _, _, info = load_model(
        models = models,
        config = config
    )
    recorder.message(info, ignore_width = True, stage = "test", write_into = "test")
    
    # test
    test_step_indicator = {}
    if (config["DDP"] == False) or (config["DDP"] == True and int(os.environ["RANK"]) == 0):
        recorder.message(f"step {0}/{len_test_dataloader - 1}", end = "\r", stage = "test", write_into = "never")
        for step, data in enumerate(test_dataloader):
            
            test_step_indicator = config["method_function"].test_step(models, data, step, config)
            recorder.message(f"step {step}/{len_test_dataloader - 1}, {dict2str(test_step_indicator)}", end = "\r", stage = "test", write_into = "test")
            
            test_step_indicator = dictionary_addition(
                A = test_step_indicator,
                B = dictionary_division(test_step_indicator, len_test_dataloader)
            )
        
    recorder.message(f"{dict2str(test_step_indicator)}", stage = "test", write_into = "test")
    
    recorder.message(f"Testing completed!", ignore_width = True)
    
    if config["DDP"] == True:
        cleanup_ddp()    
         
        
                
        
        
        
        
        
        
        
        
    