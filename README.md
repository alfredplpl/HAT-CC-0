# HAT-CC-0
HAT (Super Resolution) made of CC-0

# Getting Started
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://gist.github.com/alfredplpl/4b9134d9bdc8c2da8470a1442922bb54/hat-cc-0.ipynb)

You must install the libraries by the following code.

```shell
pip install -r requirements.txt
```

You can use this model by the following code to resize sample.jpg .

```shell
python main.py cpu sample.jpg sample_sr.png
```

I recommend use a GPU to resize your image.

```shell
python main.py cuda sample.jpg sample_sr.png
```

# Citation
```bibtex
@article{chen2022activating,
  title={Activating More Pixels in Image Super-Resolution Transformer},
  author={Chen, Xiangyu and Wang, Xintao and Zhou, Jiantao and Dong, Chao},
  journal={arXiv preprint arXiv:2205.04437},
  year={2022}
}
```