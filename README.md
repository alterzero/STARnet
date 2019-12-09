# Space-Time-Aware Multi-Resolution Video Enhancement (CVPR2020)

Project page: https://alterzero.github.io/projects/STAR.html

## Dependencies
* Python 3.5
* PyTorch >= 1.0.0
* Pyflow -> https://github.com/pathak22/pyflow
  ```Shell
  cd pyflow/
  python setup.py build_ext -i
  cp pyflow*.so ..
  ```

## Dataset
* [Vimeo-90k triplets Dataset](http://toflow.csail.mit.edu)

## Pretrained Model and Testset


## HOW TO

#Training

    ```python
    main.py
    ```

#Testing

    ```python
    eval.py
    ```

![RBPN](https://alterzero.github.io/projects/STAR.png)

## Related work
### [Deep Back-Projection Networks for Super-Resolution (CVPR2018)](https://github.com/alterzero/DBPN-Pytorch)
#### Winner (1st) of [NTIRE2018](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Timofte_NTIRE_2018_Challenge_CVPR_2018_paper.pdf) Competition (Track: x8 Bicubic Downsampling)
#### Winner of [PIRM2018](https://arxiv.org/pdf/1809.07517.pdf) (1st on Region 2, 3rd on Region 1, and 5th on Region 3)
#### Project page: https://alterzero.github.io/projects/DBPN.html
### [Recurrent Back-Projection Network for Video Super-Resolution (CVPR2019)](https://github.com/alterzero/RBPN-PyTorch)


## Citations
If you find this work useful, please consider citing it.
```
@inproceedings{STAR2020,
  title={Space-Time-Aware Multi-Resolution Video Enhancement},
  author={Haris, Muhammad and Shakhnarovich, Greg and Ukita, Norimichi},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```
