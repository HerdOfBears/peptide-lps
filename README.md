This repo contains code for performing Bayesian optimization design of peptides against rough LPS.

Lipopolysaccharide (LPS) is a bacterial endotoxin that can set off an immune cascade leading to septic shock and death. Some work has found evidence that (usually antimicrobial) peptides can (1) bind to LPS, (2) decrease cytokine production from macrophages, and (3) save mice from LPS-induced sepsis. Although it is not always the case that LPS binding is directly associated with rescuing mice. 

Here we attempt to design peptide binders for LPS. We do this by scoring binding of peptides against the lipid A core of LPS, and using score to drive a Bayesian optimization loop.

Current usage of this repo is to run umbrella sampling of peptides in serial through the use of the bash_scripts/run_process_sequences.sh script where you have an input file containing peptide IDs, sequence, and a short description (not used at the moment); an example file is included under input_files/. Separately, one could use the bayesopt_driver.py script to run simple tests of EVO_BO, an Bayesian optimization variant where an evolutionary algorithm is used to optimize the acquisition function. 

# setup
Some of the following steps are different if doing within the Digital Research Alliance of Canada (DRAC) infrastructure.

## Local / not on DRAC infra
1. Install a package management helper like uv ( [link](https://docs.astral.sh/uv/getting-started/) )
2. Clone this repo 
```
git clone [insert repo clone link]
```
3. Create a virtual environment within the the project
```
cd peptide-lps
uv venv --python 3.12
```
activate it
```
source .venv/bin/activate
```
4. Install pypi-available package dependencies using the requirements_pypi.txt file
```
uv pip install -r requirements_pypi.txt
```

5. Install pypi-unavailable packages
Additional packages unavailable on pypi need to be installed. I recommend starting by making a directory to hold these other repos.
```
mkdir additiona_repos
cd additional_repos
```
**martini_openmm**
Go to the martini openmm github ( [link](https://github.com/maccallumlab/martini_openmm) ).
```
git clone [insert repo clone link]
```
then install it
```
uv pip install martini_openmm
```
**PDBFixer** ( [link](https://github.com/openmm/pdbfixer) )
Clone it, then install. 

>[!NOTE]
> One could also do a remote repo installation by
> pip install git+https://git.example.com/MyProject.git@v1.0

6. The Weighted Histogram Analysis Method (WHAM) must be installed. 

Navigate to the github ( [link](https://github.com/agrossfield/wham) ). Clone it. The installation instructions are available in the doc.pdf, and are reproduced here:
```
cd wham
mkdir build
cd build
cmake ..
cmake --build .
```
Then
```
cmake --install .
```
to install in PATH. 

7. change the permissions of bash_scripts/setup_peptide_lps_system.sh 
```
chmod +x bash_scripts/setup_peptide_lps_system.sh
```

## On a DRAC cluster
On a DRAC cluster, like Rorqual, many of the packages may need to be loaded as a module first. OpenMM is not a pip installed package, and instead is a module that is loaded *before* activating the virtual environment. One issue you may encounter is setting up Boltz on a DRAC cluster. One method is to set it up locally somewhere, then use globus to transfer the model weights and mol/ directory to ~/.boltz (on the cluster) before starting any jobs.

# Test
To test the installation, I recommend trying to run a bayesopt_driver.py smoke test. This tests the entire umbrella sampling workflow, the Bayesian optimization loop, and the generative model. In a smoke test it sets the umbrella sampling window sim length to 100ps the number of windows to six; it also sets the number of bayesian optimization iterations to one. 




