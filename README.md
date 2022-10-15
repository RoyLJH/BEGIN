# BEGIN: Batchnorm Ensembled Generative Inversion

## Requirements
    torch==1.12.1+cu113
    torchvision==0.13.1+cu113
    
## Usage
    python inverter_rpc.py --modelnames modelA modelB modelC --devices 0 1 2
or 
    
    python inverter_fast.py --modelnames modelD modelE --devices 3 -1
    
The two scripts (`inverter_rpc.py` and `inverter_fast.py`) implements exactly same functionality.
Note that `inverter_rpc.py` is implemented by [PyTorch RPC framework](https://pytorch.org/docs/stable/rpc.html). 
