import os.path

PREDICTOR_OUTPUT_DIR = "predictor_outputs"

def get_model_path(model_name:str, tp_num:int, mem_margin:float=0.9):
    return os.path.join(PREDICTOR_OUTPUT_DIR,model_name,f"tp={tp_num}",f"mem={mem_margin}")