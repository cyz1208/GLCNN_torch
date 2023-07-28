# GLCNN with "global + local" strategy
 
The package replicates the training and testing of GLCNN model 
using carbon-based transition metal single-atom catalysts (TMSACs) or user-defined datasets.

##  Prerequisites

This package requires:

- tensorflow
- scikit-learn
- pymatgen
- networkx
- pickle
- matplotlib

The easiest way of installing the prerequisites is via [conda](https://www.anaconda.com). 
After installing `conda`, run the following command to create a new environment named `GLCNN` 
and install all prerequisites:

```bash
conda upgrade conda
conda create --name GLCNN python=3.8 -c conda-forge
```

This creates a conda environment for running GLCNN. 

Before using GLCNN, activate the environment by:

```bash
source activate GLCNN
```

After activating the environment, tensorflow 2.6 for GPU is needed:

```bash
conda install tensorflow-gpu=2.6 cudatoolkit=11.3 cudnn=8.2
```

The above three versions should be compatible with each other.

If GPU is not available, use CPU-version tensorflow instead:

```bash
pip install tensorflow
```

However, the training of GLCNN is far more efficient when using GPU.

The installation of pymatgen is as following:

```bash
conda install --channel conda-froge pymatgen
```

The other prerequisites installed via pip:

```bash
pip install scikit-learn networkx pickle matplotlib
```

Then, in directory `GLCNN`, you can test GLCNN by running:

```bash
python graph.py
python pixel.py
```

`graph.py` and `pixel.py` generate descriptor and grid inputs named `graphs.pkl` and `pixels.pkl` in `data` folder.
The user-defined structural files in VASP5.x POSCAR format (element row in file) should be 
stored in `user_catalysts` folder and named appropriately, e.g., POSCAR_1, POSCAR_2.

If the users want to train and test GLCNN using demo structures provided in `demo_catalysts` folder,
run `graph.py` and `pixel.py` as following:

```bash
python graph.py --demo
python pixel.py --demo
```

After generations of grids and descriptors, train and test GLCNN:

```bash
python GLCNN.py --batch=256 --repeat=20 --epoch=200
```
`GLCNN.py` train and test GLCNN using generated inputs `graphs.pkl` and `pixels.pkl`. 
`--batch`, `--repeat` and `--epoch` denote batch size, DA iterations and epoch to training respectively.

Using:
```bash
python GLCNN.py --help
```
to get more information.

`out_OH.csv` will be generated after GLCNN training and test, which containing predicted and true values.
The log of the GLCNN running is recorded in the `log` folder. 
The optimized GLCNN model in the training process is saved in the `model_opt` folder. 

`42`, `42_2`, `24` and `22` folders in `demo_catalysts` denote different cell expansion coefficients. 
The distribution of N outside defects in `42_2` is different from that in `42`. 
The structure of the `42` folder is as following:

```
GLCNN
├─ 42                     # cell expansion coefficient
│  ├─ 0N                  # N content outside defect
│  │  ├─ SV               # defect containing 0 N
│  │  │  ├─ Sc            # TM atom
│  │  │  │      POSCAR
│  │  │  │      POTCAR
│  │  │  ├─ Ti
│  │  │  │      POSCAR
│  │  │  │      POTCAR
│  │  │  ├─ ...
│  │  ├─ SV_1N            # defect containing 1 N
│  │  │  ├─ Sc
│  │  │  │      POSCAR
│  │  │  │      POTCAR
│  │  │  ├─ ...
```

`22`, `24` and `42_2` folders also have the same structure as `42`.
