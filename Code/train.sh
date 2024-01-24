CUDA_VISIBLE_DEVICES="0, 1" torchrun --nproc_per_node=2 train.py
watch -n 1 gpustat
watch -n 1 free -h
tmux attach -t 0 # pca test
tmux attach -t 1 # sfit train
# PCA batchSize = 16
df6147a12e4b386f97417e3acd4899d722804436
nvitop
nvidia-smi
nvcc -V
sudo apt install gpustat

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tqdm
pip install tensorboard
pip install einops
pip install opencv-python
pip install albumentations
pip install kornia
