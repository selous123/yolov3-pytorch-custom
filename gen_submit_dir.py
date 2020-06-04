### 生成提交模型到华为云modelart判分
import argparse
import shutil
import os
import torch
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--models_dir','-m', type=str, default='trained_models/yolo-v3-60e')
parser.add_argument('--submit_dir','-s', type=str, default='submit/test')
opt = parser.parse_args()

opt.submit_dir = os.path.join(opt.submit_dir, 'model')

rootPath = os.path.dirname(os.path.abspath(__file__))
## 保存内容
### 1. 复制 my_utils 目录
### 2. 复制 models.py 文件
### 3. 复制 deploy_scripts/ 文件夹

## 创建文件夹
desDir = os.path.join(rootPath, opt.submit_dir)
os.makedirs(desDir, exist_ok=True)

## 1. 转移 my_utils/
if not os.path.exists(os.path.join(desDir, 'my_utils')):
    shutil.copytree('deploy_scripts/my_utils/', os.path.join(desDir, 'my_utils'))
## 2. 保存 models.py
shutil.copy('models.py', os.path.join(desDir, 'models.py'))
## 3. 保存 deploy_scripts
shutil.copy('deploy_scripts/config.json', os.path.join(desDir, 'config.json'))
shutil.copy('deploy_scripts/customize_service.py', os.path.join(desDir, 'customize_service.py'))
shutil.copy('deploy_scripts/classify_rule.json', os.path.join(desDir, 'classify_rule.json'))
shutil.copy('deploy_scripts/train_classes.txt', os.path.join(desDir, 'train_classes.txt'))
## 4. 保存模型参数
m = torch.load(os.path.join(opt.models_dir, 'best.pt'))
torch.save(m['model'], os.path.join(desDir, 'best.pkl'))

sour_path = glob.glob(os.path.join(opt.models_dir, '*.cfg'))[0]
## 5. 复制模型参数
shutil.copy(sour_path, os.path.join(desDir, os.path.basename(sour_path)))

print("Done: copy file success!")
