# Multi-task_Survival_Analysis
This package includes three models for multi-task survival analysis.

The folder "data" includes the example data for usage. The data used for each task is stored in one single ".csv" file. Where each instance is represented as a row in file and the last two columns are survival_times and censored_indicators, respectively. 

To wrong this code, the users should first run “multi_cox_prepare.m” to generate the training and testing file from the original files. The function has two input parameter, first denote the folder where all the data stored, and the second denotes the number of cross validation folders. 

>> multi_cox_prepare '/data/Noname_addone_miRNA_use/' 3

After running this function, we will have training and testing files for each cross validation.

Now we can run the experiment for three multi-task survival analysis models.
The “example_cox_CMTL.m” reforms to the Cluster Multi-task survival analysis models. This function have 6 arguments

The Folder where the training and testing file stored
Name of training file
Name of testing file
Number of searching parameter
Number of clusters
The rate of smallest search parameter compares to the largest one 

***run example of example_cox_CMTL.m
>> example_cox_CMTL 'Noname_addone_miRNA_use/' 'train_1' 'test_1' 10 4 0.01

The “example_cox_L21.m” reforms to the L_2,1 norm regularized Multi-task survival analysis models. This function has 6 arguments

The Folder where the training and testing file stored
Name of training file
Name of testing file
Number of searching parameter  
The rate of smallest search parameter compares to the largest one 
The scale of first searching point (usually set as 1)

***run example of example_cox_CMTL.m
>> example_cox_L21 'Noname_addone_miRNA_use/' 'train_1' 'test_1' 100 0.01 1


The “example_cox_Trace.m” reforms to the Trace norm regularized Multi-task survival analysis models. This function has 6 arguments

The Folder where the training and testing file stored
Name of training file
Name of testing file
Number of searching parameter  
The rate of smallest search parameter compares to the largest one 
The scale of first searching point (usually set as 1)

***run example of example_cox_ Trace.m
>> example_cox_Trace 'Noname_addone_miRNA_use/' 'train_1' 'test_1' 100 0.01 1
