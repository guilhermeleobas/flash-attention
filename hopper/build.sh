#!/bin/bash

set -e

# Flash Attention Minimal Build Script for PHI-1 Reproducer
# Uses subshell to automatically clean up environment variables

# Run in subshell - variables are automatically cleaned up when it exits
(
    # Set minimal build flags for PHI-1 reproducer
    export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
    export FLASH_ATTENTION_DISABLE_SPLIT=TRUE
    export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
    export FLASH_ATTENTION_DISABLE_LOCAL=TRUE
    export FLASH_ATTENTION_DISABLE_CLUSTER=TRUE
    export FLASH_ATTENTION_DISABLE_VARLEN=TRUE
    export FLASH_ATTENTION_DISABLE_PACKGQA=TRUE
    export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
    export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
    export FLASH_ATTENTION_DISABLE_FP8=TRUE
    export FLASH_ATTENTION_DISABLE_FP16=FALSE
    export FLASH_ATTENTION_DISABLE_FP32=TRUE
    
    # Keep only 64-dim heads for PHI-1
    export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM128=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM256=TRUE
    
    echo "Environment variables set for minimal build..."
    
    # Install flash-attention
    python setup.py install
)