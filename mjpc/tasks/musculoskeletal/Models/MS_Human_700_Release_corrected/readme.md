# MS-Human-700 Description (MJCF)

MuJoCo Full Body Human Musculoskeletal Model

Requires MuJoCo 3.1.2 or later, for muscle force calculation.

## Overview

MS-Human-700 model is a MuJoCo model of the full-body human musculoskeletal system, with anatomically-detailed body, joint, and muscle parameters referring to the biomechanical literature. For the detailed modeling and control methods, please refer to [the paper](https://arxiv.org/abs/2312.05473).

<p float="left">
  <img src="Pictures/ms_human_738.png" width="400">
  <img src="Pictures/ms_human_738_inertia.png" width="400">
</p>

### Default Model

MS_Human_700.xml

Full body human musculoskeletal model with simple hands and torso.

Body number: 81

Joint number: 85

Muscle number: 700

### Complex Model

MS_Human_700_Complex.xml

Full body human musculoskeletal model with detailed hands.

Body number: 127

Joint number: 249

Muscle number: 738

## License

This model is released under an [Apache-2.0 License](LICENSE).

## Citation

If you use this work in an academic context, please cite the following publication:

```bibtex
@inproceedings{zuo2024self,
  title={Self model for embodied intelligence: Modeling full-body human musculoskeletal system and locomotion control with hierarchical low-dimensional representation},
  author={Zuo, Chenhui and He, Kaibo and Shao, Jing and Sui, Yanan},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={13062--13069},
  year={2024},
  organization={IEEE}
}
```