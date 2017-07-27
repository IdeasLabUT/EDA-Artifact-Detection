# Motion Artifact Detection in Wrist-Measured Electrodermal Activity Data

This repository contains the Python code and data to reproduce the experiments in our [ISWC 2017 paper "Unsupervised Motion Artifact Detection in Wrist-Measured Electrodermal Activity Data"](https://arxiv.org/abs/1707.08287). We evaluate 5 supervised and 3 unsupervised machine learning algorithms for motion artifact (MA) detection on electrodermal activity (EDA) and 3-axis accelerometer data. Our experiments are performed on two publicly available data sets:

- UT Dallas Stress (UTD) Data: 20 college students performed a sequence of tasks subjecting them to physical, cognitive, and emotional stress in a lab environment. It contains about 13 hours of data total (Birjandtalab et al., 2016). We include only the pre-processed feature matrices. The raw data is available at [http://www.utdallas.edu/~nourani/Bioinformatics/Biosensor_Data/](http://www.utdallas.edu/~nourani/Bioinformatics/Biosensor_Data/)
- Alan Walks Wales (AWW) Data: collected by Alan Dix while he walked around Wales from mid-April to July 2013. We extracted segments of data over 10 different days resulting in 10 hours of data in total (5 hours walking, 5 hours resting). We include both the feature matrices and the raw data for the 10 hours we extracted. The raw data over Alan's entire journey is available at http://alanwalks.wales/data/

Refer to the [paper](https://arxiv.org/abs/1707.08287) for more details. The code requires the [scikit-learn](http://scikit-learn.org/) Python package.

## Contents

In the root directory:

- `Feature Details.txt`: Description of all features constructed from EDA and accelerometer data.
- `FeatureMatrix_AWW.py`: Script to compute feature matrix from Alan Walks Wales raw data. This script does not need to be executed to reproduce our experiments because we also include the pre-processed feature matrix.
- `LICENSE.txt`: License for this software.
- `MachineLearning_InSample7_AWW.py`: Script to perform in-sample MA prediction (using leave-one-segment-out cross-validation) on AWW resting and walking data separately using all algorithms aside from the multi-layer Perceptron (MLP).
- `MachineLearning_InSample7_UTD.py`: Script to perform in-sample MA prediction (using leave-one-subject-out cross-validation) on UTD data using all algorithms aside from the multi-layer Perceptron (MLP).
- `MachineLearning_OutofSample7.py`: Script to perform out-of-sample MA prediction (both train on AWW/test on UTD and train on UTD/test on AWW) using all algorithms aside from the multi-layer Perceptron (MLP).
- `MLP_inSample_AWW.py`: Script to perform in-sample MA prediction (using leave-one-segment-out cross-validation) on AWW resting and walking data separately using the multi-layer Perceptron (MLP). This is the most time-consuming algorithm so we separated it into its own script.
- `MLP_InSample_UTD.py`: Script to perform in-sample MA prediction (using leave-one-subject-out cross-validation) on UTD data using the multi-layer Perceptron (MLP).
- `MLP_OutSample.py`: Script to perform out-of-sample MA prediction (both train on AWW/test on UTD and train on UTD/test on AWW) using the multi-layer Perceptron (MLP).

Each subdirectory (either `AlanWalksWales` or `UTDallas`) contains the data files. For example, for the AWW resting data, the following files are included:

- `AWW_rest_acc.csv`: Feature matrix for AWW resting data using only accelerometer features. Each row is a 5-second time window, and each column is a feature.
- `AWW_rest_all.csv`: Feature matrix for AWW resting data using all (EDA and accelerometer) features
- `AWW_rest_eda.csv`: Feature matrix for AWW resting data using only EDA features
- `AWW_rest_groups.csv`: Mapping of 5-second time windows to groups for leave-one-group-out cross-validation. The value on the *i*th row indicates which group (extracted segment of data) the *i*th second time window belongs to.
- `AWW_rest_label_All3`.csv: Labels of each time window as clean (0) or MA (1) by 3 EDA experts.
- `AWW_rest_label.csv`: Majority vote over 3 EDA expert labels for each time window.

Each file is also available for the AWW walking data and for the UTD data. For the AWW data, we also include the raw CSV files, e.g. `2013_05_14_40mins_eating.csv`, from the Affectiva Q sensor, as well as the CSV files exported from [EDA Explorer](http://eda-explorer.media.mit.edu/) containing our expert labels, e.g. `2013_05_14_40mins_eating_Epochs.csv` in the subdirectory `Raw`.

## References

Birjandtalab, J., Cogan, D., Pouyan, M. B., & Nourani, M. (2016). A non-EEG biosignals dataset for assessment and visualization of neurological status. In Proceedings of the IEEE International Workshop on Signal Processing Systems (pp. 110–114). IEEE. https://doi.org/10.1109/SiPS.2016.27

Zhang, Y., Haghdan, M., & Xu, K. S. (2017). Unsupervised motion artifact detection in wrist-measured electrodermal activity data. In Proceedings of the 21st International Symposium on Wearable Computers (to appear). Retrieved from http://arxiv.org/abs/1707.08287

## License

Distributed with a BSD license; see `LICENSE.txt`