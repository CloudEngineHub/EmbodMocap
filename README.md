<h1 align="center">[CVPR 2026] EmbodMocap: In-the-Wild 4D Human-Scene Reconstruction for Embodied Agents</h1>

<div align="center">
    <p>
        <a href="https://wenjiawang0312.github.io/">Wenjia Wang</a><sup>1,*</sup>  
        <a href="https://liangpan99.github.io/">Liang Pan</a><sup>1,*</sup>  
        <a href="https://phj128.github.io/">Huaijin Pi</a><sup>1</sup>  
        <a href="https://thorin666.github.io/">Yuke Lou</a><sup>1</sup>  
        <a href="https://xuqianren.github.io/">Xuqian Ren</a><sup>2</sup>  
        <a href="https://littlecobber.github.io/">Yifan Wu</a><sup>1</sup>  
        <br>
        <a href="https://zycliao.com/">Zhouyingcheng Liao</a><sup>1</sup>  
        <a href="http://yanglei.me/">Lei Yang</a><sup>3</sup>  
        <a href="https://rishabhdabral.github.io/">Rishabh Dabral</a><sup>4</sup>
        <a href="https://people.mpi-inf.mpg.de/~theobalt/">Christian Theobalt</a><sup>4</sup>
        <a href="https://i.cs.hku.hk/~taku/">Taku Komura</a><sup>1</sup>
    </p>
    <p>
        (*: Core Contributor)
    </p>
    <p>
        <sup>1</sup>The University of Hong Kong    
        <sup>2</sup>Tampere University
        <br>
        <sup>3</sup>The Chinese University of Hong Kong    
        <sup>4</sup>Max-Planck Institute for Informatics
    </p>

</div>

<p align="center">
    <a href="https://arxiv.org/abs/2602.23205" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-00AEEF?style=plastic&logo=arxiv&logoColor=white" alt="arXiv">
    </a>
    <a href="https://wenjiawang0312.github.io/projects/embodmocap/" target="_blank">
    <img src="https://img.shields.io/badge/Project Page-F78100?style=plastic&logo=google-chrome&logoColor=white" alt="Project Page">
    </a>
    <a href="https://www.youtube.com/watch?v=B5CDThL2ypo" target="_blank">
    <img src="https://img.shields.io/badge/YouTube Video-FF0000?style=plastic&logo=youtube&logoColor=white" alt="YouTube Video">
    </a>
    <a href="https://huggingface.co/datasets/WenjiaWang/EmbodMocap_release" target="_blank">
    <img src="https://img.shields.io/badge/🤗 HuggingFace Data-FFD21E?style=plastic&logoColor=black" alt="HuggingFace Data">
    </a>
    <a href="https://connecthkuhk-my.sharepoint.com/:f:/g/personal/wwj2022_connect_hku_hk/IgAh_tLK24aLT61TePApWqk1AdpvlVBHvyttzmO61fegoC0?e=ikzCTO" target="_blank">
    <img src="https://img.shields.io/badge/OneDrive Data-4285F4?style=plastic&logo=googledrive&logoColor=white" alt="OneDrive Data">
    </a>
</p>

<div align="center">
    <a href="https://www.youtube.com/watch?v=B5CDThL2ypo" target="_blank">
        <img src="./assets/teaser.jpg" alt="EmbodMocap Teaser" style="max-width:80%;">
    </a>
</div>

# 🗓️ News:

🎆 2026.Mar.10, we have released the code and data now, please have a try!

🎆 2026.Feb.22, EmbodMocap has been accepted to CVPR2026, codes and data will be released soon.

# 🚀 Quick Start

For new users, follow this order:

1. **Main Pipeline** - Quick downloads, preview / visualization, running the pipeline, and step-by-step workflow notes

   - [docs/embod_mocap.md](docs/embod_mocap.md)
2. **Installation** - Set up the environment, core dependencies, and manual download references

   - [docs/install.md](docs/install.md)
3. **Visualization** - Generate rendered videos or inspect scenes and motions interactively with Viser

   - [docs/visualization.md](docs/visualization.md)

Notes:

- Compared to the paper version, the open-source release replaces **PromptDA** with **LingbotDepth**.
- `fast` is mainly for users who only care about **mesh + motion** for embodied tasks.
- `standard` is for users who also need **RGBD/mask** assets for training reconstruction models.
- We provide an interactive visualization tool based on **Viser** - give it a try!

## Interactive Visualization with Viser

Our Viser-based visualization tool allows you to interactively browse scenes, sequences, and SMPL motions in 3D:

<div align="center">
    <img src="./assets/viser.gif" width="80%">
</div>

Features:

- Switch between multiple scenes and sequences
- Interactive 3D viewing of scene mesh and SMPL motion
- Real-time camera trajectory visualization
- Frame-by-frame playback control

See [docs/visualization.md](docs/visualization.md) for detailed usage.

# 🎓 Citation

If you find this project useful in your research, please consider citing us:

```
@inproceedings{wang2026embodmocap,
title = {EmbodMocap: In-the-Wild 4D Human-Scene Reconstruction for Embodied Agents.},
booktitle = {CVPR},
author = {Wang, Wenjia and Pan, Liang and Pi, Huaijin and Lou, Yuke and Ren, Xuqian and Wu, Yifan and Liao, Zhouyingcheng and Yang, Lei, Dabral, Rishabh and Theobalt, Christian and Komura, Taku},
year = {2026}
}
```

# 😁 Related Repos

We acknowledge [VGGT](https://github.com/facebookresearch/vggt), [TRAM](https://github.com/yufu-wang/tram), [ViTPose](https://github.com/ViTAE-Transformer/ViTPose), [Lang-Segment-Anything](https://github.com/luca-medeiros/lang-segment-anything), [PromptDA](https://github.com/DepthAnything/PromptDA), [Lingbot-Depth](https://github.com/Robbyant/lingbot-depth), [SAM](https://github.com/facebookresearch/segment-anything), [COLMAP](https://github.com/colmap/colmap) for their awesome codes.

# 📧 Contact

Feel free to contact me for other questions or cooperation: wwj2022@connect.hku.hk
