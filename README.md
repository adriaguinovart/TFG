# TFG

The main files and their content are described next:

- src/CrossValidation.cpp: program containing the implementation of the cross validation used to evaluate the performance for each parameter combination in the tree graph model.

- src/NestedCrossValidation.cpp: program containing the implementation of the nested cross validation used to select the best tree graph model hyper-parameters.

- src/KrigingForecast.py: implementation of the Kriging variant presented in this thesis. This file includes code from the python library PyKrige, which was partially modified so that the implemented Kriging behaves as expected. It corresponds to the two weeks training and two weeks interpolation experiment (forecast).

- src/InverseDistanceWeighting.cpp: program containing the implementation of the inverse distance weighting interpolation method.

- src/TreeForecast.cpp: program containing the implementation of the tree forecast experiment (two weeks training and two weeks interpolation).

- src/PlotNetwork.py: program used to plot the graphs using the python library NetworkX.

- src/PlotResults.py: program used to create the results plots with the data obtained in the experiments.

- src/utils/utils.h: header file that defines some auxiliar methods used in different programs.

- data/processedData.csv: file gathering the processed data used to learn the interpolation models.
