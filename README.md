# AI Enterprise Workflow Capstone
Solution for the IBM AI Enterprise Workflow Capstone project in coursera.org 

Project, data sources and deliverables are given in repository:
[https://github.com/aavail/ai-workflow-capstone](https://github.com/aavail/ai-workflow-capstone)

The solution file is Jupyter notebook "ai-workflow-capstone.ipynb"
All scripts comprising of the solution are generated while executing the notebook content.

**Part 1**
The purpose of this part of the project is ingestion and visualization of data. 
Data ingestion is performed with "data_ingestion.py" script. The data source for this project is provided in ./data directory. 
Data visualizations are generated with "data_visualization.py" script and results are stored in ./images directory.


**Part 2**
The purpose of this part of the project is data engineering and model building.
Data engineering is supported with "data_engineering.py" script. 
Model is build with "model.py" script. Script implements Stochastic Gradient, Random Forrest, Gradien Boosting and Ada Boostin regressors. 
The resulting models are stored in ./models directory.
The directory ./logs contains logs created during scripts execution.

**Part 3**
The purpose of this part of the project is to develop application accessible through API. 
Flask API is provided in "application.py" script.
Application is containerzied into Docker image using "requirements.txt" and "DockerFile" scripts.
The unit tests for the Model, API and Logs are created as package in "run-tersts.py" and ./unittest directory. 
Post production performance monitoring is based on drift analysis and is uspported with "monitor.py" script.