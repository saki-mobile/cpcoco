# CPCoCo: Localization-Free Collaborative Perception via Attentive Contrastive Consensus

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TITS-blue)](https://ieeexplore.ieee.org/document/11536884)

> **CPCoCo: Localization-Free Collaborative Perception via Attentive Contrastive Consensus**  
> Langkun Xu, Chenyang Li, Zhongxiang Zhao, Lin Wang  
> *IEEE Transactions on Intelligent Transportation Systems (TITS)

---

## Overview

<p align="center">
  <img src="docs/framework.pdf" width="90%">
</p>

---

## News

- **[2026.05]** Paper accepted by **IEEE TITS**.
- **[2026.05]** Inference code and pretrained checkpoints released.

---

## Installation

Our implementation is built upon the OpenCOOD framework.

Please follow the official installation instructions from OpenCOOD:

- OpenCOOD Installation Guide:  
  https://opencood.readthedocs.io/en/latest/md_files/installation.html

---

## Dataset Preparation

### OPV2V

Download the OPV2V dataset from:

https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V

### V2X-SIM

We use the CoAlign processed version of the V2X-SIM dataset:

https://drive.google.com/drive/folders/16_KkyjV9gVFxvj2YDCzQm1s9bVTwI0Fw

After downloading, organize the dataset following the OpenCOOD directory structure.

---

## Checkpoints

Pretrained checkpoints are available in:

```bash
opencood/logs/v2xsim_cpcoco
opencood/logs/opv2v_cpcoco
```

---

## Inference

Run inference using:

```bash
python opencood/tools/inference.py --model_dir opencood/logs/v2xsim_cpcoco --fusion_method intermediate
```

---

## Citation

If you find this work useful for your research, please cite:

```bibtex
@ARTICLE{11536884,
  author={Xu, Langkun and Li, Chenyang and Zhao, Zhongxiang and Wang, Lin},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  title={CPCoCo: Localization-Free Collaborative Perception via Attentive Contrastive Consensus},
  year={2026},
  volume={},
  number={},
  pages={1-11},
  keywords={Location awareness;Signal detection;Clouds;Modeling;Noise;Collaboration;Educational institutions;Accuracy;Vehicle-to-everything;Distance measurement;Autonomous driving;collaborative perception;intermediate fusion;localization free},
  doi={10.1109/TITS.2026.3694006}
}
```

---

## Acknowledgements

This project is built upon several excellent open-source projects.

We sincerely thank the authors of:

- OpenCOOD  
  https://github.com/DerrickXuNu/OpenCOOD

- CoAlign  
  https://github.com/yifanlu0227/CoAlign

for their outstanding contributions to the collaborative perception community.

---

## License

This project is released under the Apache 2.0 License.
