# log_completion
 
The file "train_station.py" contains the main function for training and settings of several training hyperparameters.

The file "test_cat_sam.ipynb" is used to test the prediction performance of the FlexLogNet model.

The "utils" folder contains the "load_data.py" file, which generates training samples based on real data, and constructs connections between well logs based on the missingness of well logs in the training samples. These connections are used for message passing between nodes in the graph neural network (GNN). The "model.py" file constructs the FlexLogNet network. The "utils.py" file contains the network training process, including visualization of the training loss curve and adaptive learning rate. Other files are used for visualization and some post-processing calculations, such as the computation of the evaluation metric Pearson correlation coefficient (PCC).

The "Preprocessing" folder contains programs for pre-processing operations, including median filtering, standardization, normalization, etc.

Both Please follow this link https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html to configure the environment. Please select torch_geometric version 2.1.0 or later
