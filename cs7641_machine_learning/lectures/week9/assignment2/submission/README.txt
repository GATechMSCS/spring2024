The files to access the code can be found here: https://gatech.app.box.com/folder/248617736940. Before running, it's important to note that the models take over 25 hours to run.

The technologies used in this paper:
1) python 3.11.7
2) pandas 2.1.4
3) jupyter 1.0.0
4) ipython 8.20.0
5) matplotlib 3.8.0
6) numpy 1.26.3 
7) seaborn 0.12.2
8) mlrose-hiive 2.2.4

Step to replicate results:
1) If Conda is not already installed, you can download here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
2) In the zip file you'll find CS7641_A2_ScottSchmidl.yml
3) From the terminal run: conda env create --file scottschmidl_A2.yml
4) From the terminal run: conda activate scottschmidl_A2
5) Create a directory, for example scott_schmidl_A2
6) From the terminal cd into that directory
7) From the terminal run: git clone https://github.com/hiive/mlrose.git
8) From the terminal run: pip install ./mlrose
9) You should now be able to run the notebook named assignment2.ipynb.
