# MAESTRO: An Open-source Infrastructure for Modeling Dataflows within Deep Learning Accelerators
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

# What is MAESTRO?
MAESTRO is an open-source tool for modeling and evaluating the performance and energy-efficiency of different dataflows. MAESTRO is actively developed by the [Synergy Lab](https://synergy.ece.gatech.edu/) at [Georgia Institute of Technology](https://www.gatech.edu/). For more details about MAESTRO, please visit the following links.

- [MAESTRO Website](http://maestro.ece.gatech.edu/)
- [MAESTRO Docs](http://maestro.ece.gatech.edu/docs/build/html/index.html)

# Notices
### May 26th, 2021

We updated the hardware description file, added off-chip bandwidth added as constraint.

We added a validation folder with data for Eyeriss and MAERI from MICRO 2019 paper.

### Oct 13th, 2020

We added a direct support for GEMM layers. For more information, please take a look at [here](http://maestro.ece.gatech.edu/docs/build/html/layer_supported.html).

### May 13th, 2020

We updated the naming convention of mappings and the directory structure of data folder.

### Oct 14th, 2019

Latest codebase released along with MAESTRO MICRO 2019 paper.

### Citation ###
```
@inproceedings{maestro_micro2019,
  author    = {Hyoukjun Kwon and
               Prasanth Chatarasi and
               Michael Pellauer and
               Angshuman Parashar and
               Vivek Sarkar and
               Tushar Krishna},
  title     = {Understanding Reuse, Performance, and Hardware Cost of {DNN} Dataflow:
               {A} Data-Centric Approach},
  booktitle = {Proceedings of the 52nd Annual {IEEE/ACM} International Symposium
               on Microarchitecture, {MICRO}},
  pages     = {754--768},
  publisher = {{ACM}},
  year      = {2019},
}

```

 # Maintainers
- Felix (Sheng-Chun) Kao (felix@gatech.edu)
- Geonhwa Jeong (geonhwa.jeong@gatech.edu)
- Tushar Krishna (tushar@ece.gatech.edu)


# Technical Contributors
- Hyoukjun Kwon (Georgia Tech, now at Facebook Reality Labs): Main developer (core framework and functionalities)
- Prasanth Chatarasi (Georgia Tech, now at IBM Research): APIs + interface to mapping optimizers.
- Felix (Sheng-Chun) Kao (Georgia Tech): Pytorch frontend + updates to cost-model/interface + GAMMA mapper
- Geonhwa Jeong (Georgia Tech): Keras frontend + debugging + website maintainer.
- Saurabh Malik (Georgia Tech, now at Microsoft): Jupyter Notebooks demo + website.
