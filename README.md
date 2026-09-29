# CFS-BMF Design & Response Prediction Platform
This platform was developed by the AXIS Laboratory, Department of Architectural Engineering, Gachon University, for the design and response prediction of cold-formed steel bolted moment frames (CFS-BMFs).
AXIS Lab: https://sites.google.com/view/axislab



## Running the Platform
After activating an existing Python virtual environment, run the following command in the project directory:

```powershell
python run_platform.py
```
`run_platform.py` checks the required Python packages and launches the Streamlit-based platform.



## Workflow
The platform consists of the following steps:

1. Design Screening
2. Candidate Selection & Editing
3. ML Critical Response Point Prediction
4. DL Full Response Prediction
5. Design Report

When the platform is launched, a landing page displaying the AXIS Lab logo is shown first.



## Repository Structure
An example of the overall project structure is shown below:

CFS_BMF_response_prediction/
├── assets/
├── data/
│   └── dataset.mat
├── models/
│   ├── crp_models_joblib/
│   └── BCRP_script.pt
├── BCRP_dataset_X.pkl
├── BCRP_dataset_Y.pkl
├── run_platform.py
└── README.md



## Database
The large `.mat` database file is not included in the GitHub repository because of its file size.

Database download:  
[[Download link](https://u.pcloud.link/publink/show?code=kZwvi4JZEBGKqbdvh8JKarWqJIoeUbXD1uCX)]

After downloading the database, save it at:
The program automatically detects `database.mat` or other `.mat` files located in the `data/` directory.




## Candidate Selection & Editing
Selected candidate designs can be modified directly within the platform.
The editing function is restricted to the allowable range displayed below each design variable. Values outside the specified range cannot be entered or used for prediction.




## ML Models
The `crp_models_joblib` directory is not included in the GitHub repository because of its file size.

ML model download:  
[[Download link](https://u.pcloud.link/publink/show?code=kZwvi4JZEBGKqbdvh8JKarWqJIoeUbXD1uCX)]

After downloading `crp_models_joblib`, save the directory at:




## DL Model
The following files are included in the repository:

`BCRP_dataset_X.pkl`
`BCRP_dataset_Y.pkl`
`BCRP_script.pt`
can be prepared once before the first use of the DL prediction module using one of the following methods:

1. Place the file directly at `models/BCRP_script.pt`.
2. If an existing `BCRP_script.pt` file is available in a parent research directory, `run_platform.py` will automatically search for and copy it.
3. Select and save the file using the **First-time DL setup** option in the platform.

Once the file has been saved, it does not need to be uploaded again in subsequent runs.




## Citation
If you use this platform, its associated database, or the prediction models in your research, please cite the related publication:
[Paper information]




## Contact
AXIS Lab  
Department of Architectural Engineering  
Gachon University  
https://sites.google.com/view/axislab