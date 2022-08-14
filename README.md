<<<<<<< HEAD
# GDAMF
Code for the paper "Cost-effective Framework for Gradual Domain Adaptation with Multifidelity".<br>
https://arxiv.org/abs/2202.04359


# Dataset instructions
Each dataset can be downloaded from the following links.

- MNIST dataset is downloaded by pytorch

- The portraits dataset can be downloaded from
https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=0

- The gas sensor dataset can be downloaded from
http://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset+at+Different+Concentrations#

- The cover type dataset can be downloaded from
https://archive.ics.uci.edu/ml/datasets/covertype


After downloading, extract the file, and place them as follows.

data <br>
&nbsp; ├─ gas sensor ─ batch*.dat files<br>
&nbsp; ├─ cover type ─ covtype.data<br>
&nbsp; ├─ portraits <br>
&emsp;&emsp;&emsp;&emsp; ├─ F ─ png files<br>
&emsp;&emsp;&emsp;&emsp; ├─ M ─ png files<br>


# Usage
Please refer to the gdamf_env.yml for the libraries used.<br>
If necessary, create an environment with the following command.
>  conda env create -n your-env-name -f gdamf_env.yml

Run the following command to obtain experiment.
> python main.py  mnist

Run the following command to obtain the result of baseline methods.
> python competitor.py mnist dsaoda

We use AWS (g4dn.8xlarge instance) for above experiment, and the above script takes about a day to run.


=======
# gdamf
>>>>>>> commit test 2
