import torch
import flash_attn_interface as F

# monkey-patch flash_attn_interface to torch.compile flash_attn_*
F.flash_attn_func = torch.compile(F.flash_attn_func, backend="eager")
F.flash_attn_varlen_func = torch.compile(F.flash_attn_varlen_func, backend="eager")

from test_flash_attn import *
