# ETGPDA
> Code and Datasets for "Identification of piRNA-disease association based on embedding transformation graph convolutional network" 
## Datasets
- data/piRNA-disease.csv is the piRNA_disease association matrix, which contain 5002 associations between 4350 piRNAs and 21 diseases.

- data/SimRNA.csv is the piRNA similarity matrix of 4350 piRNAs,which is calculated based on piRNA sequence features.

- data/SimDisease.csv is the disease similarity matrix of 21 diseases,which is calculated based on disease mesh descriptors.
## Code
### Environment Requirement
The code has been tested running under Python 3.6.8. The required packages are as follows:
- numpy == 1.15.4
- scipy == 1.1.0
- tensorflow == 1.12.0
### Usage
```shell
python main-main.py
```