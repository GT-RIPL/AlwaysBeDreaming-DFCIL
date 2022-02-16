## Always Be Dreaming: A New Approach for Data-Free Class-Incremental Learning
PyTorch code for the ICCV 2021 paper:\
**Always Be Dreaming: A New Approach for Data-Free Class-Incremental Learning**\
**_[James Smith]_**, Yen-Chang Hsu, [Jonathan Balloch], Yilin Shen, Hongxia Jin, [Zsolt Kira]\
International Conference on Computer Vision (ICCV), 2021\
[[arXiv]] [[pdf]] [[project]]

<p align="center">
<img src="ABD_Diagram.png" width="100%">
</p>

## Abstract
Modern computer vision applications suffer from catastrophic forgetting when incrementally learning new concepts over time. The most successful approaches to alleviate this forgetting require extensive replay of previously seen data, which is problematic when memory constraints or data legality concerns exist. In this work, we consider the high-impact problem of Data-Free Class-Incremental Learning (DFCIL), where an incremental learning agent must learn new concepts over time without storing generators or training data from past tasks. One approach for DFCIL is to replay synthetic images produced by inverting a frozen copy of the learner's classification model, but we show this approach fails for common class-incremental benchmarks when using standard distillation strategies. We diagnose the cause of this failure and propose a novel incremental distillation strategy for DFCIL, contributing a modified cross-entropy training and importance-weighted feature distillation, and show that our method results in up to a 25.1% increase in final task accuracy (absolute difference) compared to SOTA DFCIL methods for common class-incremental benchmarks. Our method even outperforms several standard replay based methods which store a coreset of images.

## Installation

### Prerequisites
* python == 3.6
* torch == 1.0.1
* torchvision >= 0.2.1

### Setup
 * Install anaconda: https://www.anaconda.com/distribution/
 * set up conda environmet w/ python 3.6, ex: `conda create --name py36 python=3.6`
 * `conda activate py36`
 * `sh install_requirements.sh`

### Datasets
Download/Extract the following datasets to the dataset folder under the project root directory.
* For CIFAR-10 and CIFAR-100, download the python version dataset [here][cifar].

## Training
All commands should be run under the project root directory.

```bash
sh experiments/cifar100-fivetask.sh # tables 1,2
sh experiments/cifar100-tentask.sh # tables 1,2
sh experiments/cifar100-twentytask.sh # tables 1,2
```

## Results
Results are generated for various task sizes. See the main text for full details. Numbers represent final accuracy in three runs (higher the better). 

### CIFAR-100 (no coreset)
tasks | 5 | 10 | 20
--- | --- | --- | ---
UB | 69.9 ± 0.2 | 69.9 ± 0.2 | 69.9 ± 0.2
Base | 16.4 ± 0.4 | 8.8 ± 0.1 | 4.4 ± 0.3
LwF | 17.0 ± 0.1 | 9.2 ± 0.0 | 4.7 ± 0.1
LwF.MC | 32.5 ± 1.0 | 17.1 ± 0.1 | 7.7 ± 0.5
DGR | 14.4 ± 0.4 | 8.1 ± 0.1 | 4.1 ± 0.3
DeepInversion | 18.8 ± 0.3 | 10.9 ± 0.6 | 5.7 ± 0.3
Ours | 43.9 ± 0.9 | 33.7 ± 1.2 | 20.0 ± 1.4

### CIFAR-100 (with 2000 image coreset)
tasks | 5 | 10 | 20
--- | --- | --- | ---
UB | 69.9 ± 0.2 | 69.9 ± 0.2 | 69.9 ± 0.2
Naive Rehearsal | 34.0 ± 0.2 | 24.0 ± 1.0 | 14.9 ± 0.7
LwF | 39.4 ± 0.3 | 27.4 ± 0.8 | 16.6 ± 0.4
E2E | 47.4 ± 0.8 | 38.4 ± 1.3 | 32.7 ± 1.9
BiC | 53.7 ± 0.4 | 45.9 ± 1.8 | 37.5 ± 3.2
Ours (no coreset) | 43.9 ± 0.9 | 33.7 ± 1.2 | 20.0 ± 1.4

## Acknowledgement
This work is supported by Samsung Research America.

## Citation
If you found our work useful for your research, please cite our work:

    @article{smith2021always,
      author    = {Smith, James and Hsu, Yen-Chang and Balloch, Jonathan and Shen, Yilin and Jin, Hongxia and Kira, Zsolt},
      title     = {Always Be Dreaming: A New Approach for Data-Free Class-Incremental Learning},
      booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
      month     = {October},
      year      = {2021},
      pages     = {9374-9384}
    }

[James Smith]: https://jamessealesmith.github.io/
[Jonathan Balloch]: https://jballoch.com/
[Zsolt Kira]: https://www.cc.gatech.edu/~zk15/
[arXiv]: https://arxiv.org/abs/2106.09701
[pdf]: https://openaccess.thecvf.com/content/ICCV2021/papers/Smith_Always_Be_Dreaming_A_New_Approach_for_Data-Free_Class-Incremental_Learning_ICCV_2021_paper.pdf
[project]: https://jamessealesmith.github.io/project/dfcil/
[cifar]: https://www.cs.toronto.edu/~kriz/cifar.html
