import os
import sys
import subprocess
import itertools

# 定义参数网格
# 重点调优：L0权重(去噪力度) 和 En-CLU强度
grids = {
    'l0_weight': [1e-5, 5e-6],           # 必须很小
    'lambda_en_clu': [5.0, 10.0, 15.0],  # 引导强度
    'conf_start': [0.5, 0.6],            # 初始门槛
    'lr': [1e-4, 2e-4]
}

keys = list(grids.keys())
combinations = list(itertools.product(*grids.values()))

print(f"Total configs to test: {len(combinations)}")
best_acc = 0
best_cfg = None

for i, vals in enumerate(combinations):
    cfg = dict(zip(keys, vals))
    print(f"\n[{i+1}/{len(combinations)}] Testing: {cfg}")
    
    cmd = [sys.executable, 'train_fusion_advanced.py']
    for k, v in cfg.items():
        cmd.extend([f'--{k}', str(v)])
        
    try:
        # 运行训练
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200) # 20分钟超时
        
        # 解析结果
        acc = 0.0
        for line in result.stdout.split('\n'):
            if "FINAL_RESULT" in line:
                acc = float(line.split('ACC:')[1].strip())
        
        print(f"  -> Result ACC: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            print(f"  -> New Best!")
            
    except Exception as e:
        print(f"  -> Error: {e}")

print("\n" + "="*50)
print(f"BEST CONFIG: {best_cfg}")
print(f"BEST ACC: {best_acc:.4f}")
print("="*50)