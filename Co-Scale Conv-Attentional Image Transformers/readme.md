## Co-Scale Conv-Attentional Image Transformers

### Note

If you have a better implementation or find a bug, please let me know 😆 , thanks!

- 2021-04-23 The author release the code of CoaT. Official code is preferred. <https://github.com/mlpc-ucsd/CoaT/blob/main/src/models/coat.py>. In my implementation, some details are different with the author's. For example:
  - I did not introduce the CLS token for the classification task, which would cause the difference in the specific calculation forms of the attention and the convolutional relative position encoding. 
  - I have only implemented the form in which all the attention heads have the same window size.

### Change Log

- 2021-04-18 Change the method `tensor.unflatten()` to `einops.rearrange()` for avoiding some problems.
- 2021-04-15 Create the initial version of Conv-Attention.
- 2021-04-16 Modify the `Conditional Positional Encodings for Vision Transformers` to depthwise convolution.

### Conv-Attentional

![image](https://user-images.githubusercontent.com/26847524/114978283-9d4bde00-9ebb-11eb-9229-565ba41d59c6.png)

![image](https://user-images.githubusercontent.com/26847524/114978366-b3599e80-9ebb-11eb-8bb9-de1ee6ae696e.png)

### Project

- https://github.com/mlpc-ucsd/CoaT

### bibtex

```
@misc{xu2021coscale,
      title={Co-Scale Conv-Attentional Image Transformers},
      author={Weijian Xu and Yifan Xu and Tyler Chang and Zhuowen Tu},
      year={2021},
      eprint={2104.06399},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
