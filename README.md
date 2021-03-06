<div align="center">

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5"></a>

# Ellipse R-CNN

</div>

A PyTorch (Lightning) implementation of Ellipse R-CNN. Originally developed for [another project](https://github.com/wdoppenberg/crater-detection), it has proven succesful in predicting instanced ellipses.
The methodology is based on [this paper](https://arxiv.org/abs/2001.11584), albeit different in the sense that this model uses the regressed bounding box predictions instead of region proposals as the base for predicted normalised ellipse parameters.
