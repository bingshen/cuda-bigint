# cuda-bigint

目的是实现基于cuda的加密算法，批量加密提升速度

# 能力
目前已经能够在GPU当中正常执行大整数的加、减、乘、除、取余、比较、高次幂取余、位移等操作

并且实现了进制转换和字符串初始化以及按位随机生成大整数的方法

目前的主要经历是在优化各个算法的执行速度

结合GMP实现了大质数的生成方法，并可以初始化RSA的加密公钥私钥对

# 性能

# NVIDIA GeForce RTX 4060 Laptop GPU

random cost:0.933009

AoS add total_cost:3.2006s

SoA add total_cost:2.20597s

AoS subtract total_cost:3.1308s

SoA subtract total_cost:2.31791s

AoS multiply total_cost:10.6855s

SoA multiply total_cost:7.7833s

random cost:4.94042

AoS divide total_cost:11.4038s

SoA divide total_cost:7.67827s

AoS mod total_cost:11.7484s

SoA mod total_cost:9.35145s

random cost:4.05884

AoS power-mod total_cost:191.225s

SoA power-mod total_cost:155.377s

random cost:3.58363

AoS encrypt_total:1.56357s

AoS decrypt_total:158.047s

AoS total_cost:159.613s

SoA encrypt_total:1.12171s

SoA decrypt_total:143.849s

SoA total_cost:144.973s




# Intel(R) Xeon(R) Gold 6136 CPU @ 3.00GHz   3.00 GHz

add cost time:13.173848017636292s

subtract cost time:8.430511683475782s

multiply cost time:167.16211960807226s

div cost time:278.7772124865172s

mod cost time:266.27461345036033s

power_mod cost time:2973.2182057661344s

power_mod cost time:49.02678179423386s

power_mod cost time:2750.914678764818s
