#!/bin/bash

__C=256
__K=512
__W=246
__H=246
__R=11
__S=11
__SIGMAW=1
__SIGMAH=1

__C_B=64
__K_B=64
__W_B=123
__H_B=2
__RP_B=11
__RPP_B=1
__SP_B=11
__SPP_B=1

./measure $__C $__K $__W $__H $__R $__S $__SIGMAW $__SIGMAH $__C_B $__K_B $__W_B $__H_B $__RP_B $__RPP_B $__SP_B $__SPP_B;

