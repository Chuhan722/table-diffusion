"""归因 GSD 每 32,561 代一次的 ~25 s 突发：原版循环把每代 tell() 返回的设备标量 rep_best 存进 Python 列表，
窗口末 jnp.array(strategy_rewards) 把 32,561 个设备标量逐个拉回主机。本脚本复现该操作并计时。
用法（远端 aim 环境、空闲 GPU）：CUDA_VISIBLE_DEVICES=<uuid> JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/miniconda3/envs/aim/bin/python logs_eps/effort/gsd_burst_probe.py
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

N = 32561


@jax.jit
def tell_like(x):
    return x > 0.5


key = jax.random.PRNGKey(0)
rewards = []
t0 = time.time()
for i in range(N):
    rep_best = tell_like(jax.random.uniform(jax.random.fold_in(key, i)))  # 每代一个新的设备标量
    rewards.append([i % 4, rep_best])
jax.block_until_ready(rewards[-1][1])
t1 = time.time()
arr = jnp.array(rewards)
jax.block_until_ready(arr)
t2 = time.time()
print(f"设备 {jax.devices()[0].platform}；造 {N} 个设备标量 {t1 - t0:.1f} s；jnp.array(列表) {t2 - t1:.1f} s；形状 {arr.shape}")
host = [[i % 4, bool(np.asarray(r))] for i, (_, r) in enumerate(rewards)]
t3 = time.time()
arr2 = jnp.array(host)
jax.block_until_ready(arr2)
print(f"对照：逐个 np.asarray 拉回 {t3 - t2:.1f} s，再 jnp.array(主机列表) {time.time() - t3:.2f} s")
