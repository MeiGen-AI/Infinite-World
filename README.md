<h1 align="center">Infinite-World</h1>

<h3 align="center">Scaling Interactive World Models to 1000-Frame Horizons via Pose-Free Hierarchical Memory</h3>

<p align="center">
  <a href="http://arxiv.org/abs/2602.02393"><img src="https://img.shields.io/badge/arXiv-2602.02393-b31b1b.svg" alt="arXiv"></a>
  <a href="https://rq-wu.github.io/projects/infinite_world"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
</p>

<p align="center">
  <strong>Ruiqi Wu</strong><sup>1,2,3*</sup>, <strong>Xuanhua He</strong><sup>4,2*</sup>, <strong>Meng Cheng</strong><sup>2*</sup>, <strong>Tianyu Yang</strong><sup>2</sup>, <strong>Yong Zhang</strong><sup>2‡</sup>, <strong>Chunle Guo</strong><sup>1,3†</sup>, <strong>Chongyi Li</strong><sup>1,3</sup>, <strong>Ming-Ming Cheng</strong><sup>1,3</sup>
</p>

<p align="center">
  <sup>1</sup>Nankai University &nbsp; <sup>2</sup>Meituan &nbsp; <sup>3</sup>NKIARI &nbsp; <sup>4</sup>HKUST
</p>

<p align="center">
  <sup>*</sup>Equal Contribution &nbsp; <sup>†</sup>Corresponding Author &nbsp; <sup>‡</sup>Project Leader
</p>

<p>&nbsp;</p>
<p align="center">
  <a href="https://www.youtube.com/watch?v=4ZJrXpAMNbU"><img src="assets/title.png" alt="Demo" width="640"></a>
</p>
<p>&nbsp;</p>


---

## Highlights

**Infinite-World** is a robust interactive world model with:

- **Real-World Training** — Trained on real-world videos without requiring perfect pose annotations or synthetic data
- **1000+ Frame Memory** — Maintains coherent visual memory over 1000+ frames via Hierarchical Pose-free Memory Compressor (HPMC)
- **Robust Action Control** — Uncertainty-aware action labeling ensures accurate action-response learning from noisy trajectories

<p align="center">
  <img src="./assets/framework.png" alt="Infinite-World Framework" width="100%">
</p>


## Installation

**Environment:** Python 3.10, CUDA 12.4 recommended.

### 1. Create conda environment

```bash
conda create -n infworld python=3.10
conda activate infworld
```

### 2. Install PyTorch with CUDA 12.4

Install from the official PyTorch index (no local whl):

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```


### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Checkpoint Configuration

All model paths are configured in **`configs/infworld_config.yaml`**. Paths are relative to the project root unless absolute.

### Download checkpoints

Download from [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) and place files under `checkpoints/`:

| File / directory | Config key | Description |
|------------------|------------|-------------|
| `models/Wan2.1_VAE.pth` | `vae_cfg.vae_pth` | VAE weights |
| `models/models_t5_umt5-xxl-enc-bf16.pth` | `text_encoder_cfg.checkpoint_path` | T5 text encoder |
| `models/google/umt5-xxl` (folder) | `text_encoder_cfg.tokenizer_path` | T5 tokenizer |
| `infinite_world_model.ckpt` | `checkpoint_path` | DiT model weights |



- **DiT checkpoint:** Can be downloaded from [TBD]().

---

## Upload to Hugging Face (including checkpoints)

To upload this repo to Hugging Face Hub (code + `checkpoints/`):

1. **Login**
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```
   Use a token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (need write permission).

2. **Upload**
   From the project root (`infinite-world/`):
   ```bash
   python scripts/upload_to_hf.py YOUR_USERNAME/infinite-world
   ```
   Or set the repo and run:
   ```bash
   export HF_REPO_ID=YOUR_USERNAME/infinite-world
   python scripts/upload_to_hf.py
   ```

   The script uploads the whole directory (including `checkpoints/`) and skips `__pycache__`, `outputs`, `.git`, etc. Large checkpoint files are uploaded via the Hub API; the first run may take a while depending on size and network.

3. **Create repo manually (optional)**  
   You can create the model repo first at [https://huggingface.co/new](https://huggingface.co/new) (type: **Model**), then run the script with that `repo_id`.

---

## Results

### Quantitative Comparison

| Model | Mot. Smo.↑ | Dyn. Deg.↑ | Aes. Qual.↑ | Img. Qual.↑ | Avg. Score↑ | Memory↓ | Fidelity↓ | Action↓ | ELO Rating↑ |
|:------|:----------:|:----------:|:-----------:|:-----------:|:-----------:|:-------:|:---------:|:-------:|:-----------:|
| Hunyuan-GameCraft | 0.9855 | 0.9896 | 0.5380 | 0.6010 | 0.7785 | 2.67 | 2.49 | 2.56 | 1311 |
| Matrix-Game 2.0 | 0.9788 | **1.0000** | 0.5267 | **0.7215** | 0.8068 | 2.98 | 2.91 | 1.78 | 1432 |
| Yume 1.5 | 0.9861 | 0.9896 | **0.5840** | <u>0.6969</u> | **0.8141** | <u>2.43</u> | <u>1.91</u> | 2.47 | 1495 |
| HY-World-1.5 | **0.9905** | **1.0000** | 0.5280 | 0.6611 | 0.7949 | 2.59 | 2.78 | **1.50** | <u>1542</u> |
| **Infinite-World** | <u>0.9876</u> | **1.0000** | <u>0.5440</u> | <u>0.7159</u> | <u>0.8119</u> | **1.92** | **1.67** | <u>1.54</u> | **1719** |


## Citation

If you find this work useful, please consider citing:

```bibtex
@article{wu2026infiniteworld,
  title={Infinite-World: Scaling Interactive World Models to 1000-Frame Horizons via Pose-Free Hierarchical Memory},
  author={Wu, Ruiqi and He, Xuanhua and Cheng, Meng and Yang, Tianyu and Zhang, Yong and Kang, Zhuoliang and Cai, Xunliang and Wei, Xiaoming and Guo, Chunle and Li, Chongyi and Cheng, Ming-Ming},
  journal={arXiv preprint arXiv:2602.02393},
  year={2026}
}
```


## License

This project is released under the [MIT License](LICENSE).
