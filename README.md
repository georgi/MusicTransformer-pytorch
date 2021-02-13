# Music Transformer: Generating Music with Long-Term Structure

- 2019 ICLR, Cheng-Zhi Anna Huang, Google Brain
- Re-producer : Yang-Kichang
- [paper link](https://arxiv.org/abs/1809.04281) 
- [paper review](https://github.com/SSUHan/PaparReviews/issues/13)


## Contribution

* Domain: Dramatically reduces the memory footprint, allowing it to scale to musical sequences on the order of minutes.
* Algorithm: Reduced space complexity of Transformer from O(N^2D) to O(ND).



## Preprocessing

* In this repository using single track method (2nd method in paper.).

* If you want to get implementation of method 1, see [here](https://github.com/COMP6248-Reproducability-Challenge/music-transformer-comp6248) .


## Prepare Dataset	

```bash
$ python preprocess.py {midi_load_dir} {dataset_save_dir}
```


## Hyper Parameter

* learning rate : 0.0001
* head size : 4
* number of layers : 6
* seqence length : 2048
* embedding dim : 256 (dh = 256 / 4 = 64)
* batch size : 2




## Generate Music

```bash
$ python generate.py -c {config yml file 1} {config yml file 2} -m {model_dir}
```


