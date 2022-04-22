### UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection
Andra Acsintoae*, Andrei Florescu*, Mariana-Iuliana Georgescu*, Tudor Mare*, Paul Sumedrea*, Radu Tudor Ionescu, Fahad Shahbaz Khan, Mubarak Shah

*equal contribution

IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

Official URL: TBA

ArXiv URL: https://arxiv.org/pdf/2111.08644.pdf


We present an abnormal video in the left side (the anomalous regions are emphasised with red contour) while in the right side we present a normal video from the UBnormal data set.

![](./imgs/abnormal_scene_29_scenario_3.gif)
![](./imgs/normal_scene_16_scenario_2.gif)

# 🌟 NEW: We released the ground-truth for the test set. 

### Table of Contents:

  [Description of UBnormal](#description) 
  
  [Download data set](#download)
  
  [Statistics](#statistics)
  
  [State-of-the-art results](#State-of-the-art-results)
  
  [Scripts](#scripts)
  
  [License](#license)
  
  [Citation](#citation) 
 

### Description
UBnormal is a new supervised open-set benchmark composed of multiple virtual scenes for video anomaly detection. 
Unlike existing data sets, we introduce abnormal events annotated at the pixel level at training time,
for the first time enabling the use of fully-supervised learning methods for abnormal event detection. 
To preserve the typical open-set formulation, we make sure to include disjoint sets of anomaly types in our training
and test collections of videos.

Examples of actions from our data set:    
<img src="https://raw.githubusercontent.com/lilygeorgescu/UBnormal/main/imgs/ubnormal_examples.png" width="500">

### Download
The UBnormal data set can be downloaded from [here](https://drive.google.com/file/d/1KbfdyasribAMbbKoBU1iywAhtoAt9QI0/view?usp=sharing).
  
### Statistics 
<img src="https://raw.githubusercontent.com/lilygeorgescu/UBnormal/main/imgs/statistics.png" width="400">

### State-of-the-art results
The results on the UBnormal data set on the test set:
<table>
<tr>
    <td>Method</td> 
    <td>Micro-AUC</td>
    <td>Macro-AUC</td>
    <td>RBDC</td>
    <td>TBDC</td>
</tr>
  
<tr>
    <td>Georgescu et al. [1] + UBnormal anomalies</td> 
    <td>61.3</td>
    <td>85.6</td>
    <td>25.430</td>
    <td>56.272</td>
</tr>

<tr>
    <td>Sultani et al. [2] (fine-tuned)</td> 
    <td>50.3</td>
    <td>76.8</td>
    <td>0.002</td>
    <td>0.001</td>
</tr>

<tr>
    <td>Bertasius et al. [3] (1/32 sample rate, fine-tuned)</td> 
    <td>68.5</td>
    <td>80.3</td>
    <td>0.041</td>
    <td>0.053</td>
</tr>

</table>

<div>
<sub>
[1] M.I. Georgescu, R.T. Ionescu, F.S. Khan, M. Popescu, and M. Shah. A Background Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video. TPAMI 2021
</sub>
</div>
<div>
<sub>
[2] W. Sultani, C. Chen, and M. Shah. Real-World Anomaly Detection in Surveillance Videos. CVPR 2018
</sub>
</div>
<div>
<sub>
[3] G. Bertasius, H. Wang, and L. Torresani. Is Space-Time Attention All You Need for Video Understanding. ICML 2021
</sub>
</div>


## Scripts

See the [scripts](https://github.com/lilygeorgescu/UBnormal/tree/main/scripts) folder.

## License
The present data set is released under the 
Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license.

## Citation 
Please cite our work if you use any material released in this repository.
```
@InProceedings{Acsintoae_CVPR_2022,
  author    = {Andra Acsintoae and Andrei Florescu and Mariana{-}Iuliana Georgescu and Tudor Mare and  Paul Sumedrea and Radu Tudor Ionescu and Fahad Shahbaz Khan and Mubarak Shah},
  title     = {UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2022},
  }
``` 
