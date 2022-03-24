# VAPO

[**Affordance Learning from Play for Sample-Efficient Policy Learning**](https://arxiv.org/pdf/2203.00352.pdf)

Jessica Borja, [Oier Mees](https://www.oiermees.com/), [Gabriel Kalweit](https://nr.informatik.uni-freiburg.de/people/gabriel-kalweit), [Lukas Hermann](http://www2.informatik.uni-freiburg.de/~hermannl/), [Joschka Boedecker](https://nr.informatik.uni-freiburg.de/people/joschka-boedecker), [Wolfram Burgard](http://www2.informatik.uni-freiburg.de/~burgard)

[ICRA 2022](https://www.icra2022.org/)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


We present Visual Affordance-guided Policy Optimization (VAPO), a novel approach that extracts a selfsupervised visual affordance model from human teleoperated
play data and leverages it to enable efficient policy learning and motion planning. We combine model-based planning with model-free deep reinforcement learning (RL) to learn policies
that favor the same object regions favored by people, while requiring minimal robot interactions with the environment.
We find that our policies train 4x faster than the baselines and generalize better to novel objects because our visual affordance model can
anticipate their affordance regions. More information at our [project page](http://vapo.cs.uni-freiburg.de/).

<p align="center">
  <img src="http://vapo.cs.uni-freiburg.de/images/motivation.png" width="75%"/>
</p>

## Installation
- [Install locally](./docs/local_setup.md)
- [Install using Docker](./docs/docker_setup.md)

## Usage
- [Affordance model](./docs/affordance.md)
- [Policy](./docs/policy.md)

Implementation of Visual Affordance-guided Policy Optimization


## Citation

If you find the dataset or code useful, please cite:

```
@inproceedings{borja22icra,
author = {Jessica Borja-Diaz and Oier Mees and Gabriel Kalweit and  Lukas Hermann and Joschka Boedecker and Wolfram Burgard},
title = {Affordance Learning from Play for Sample-Efficient Policy Learning},
booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation  (ICRA)},
year = 2022,
address = {Philadelphia, USA}
}
```

## License

MIT License
