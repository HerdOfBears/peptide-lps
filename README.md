This repo contains code for performing Bayesian optimization design of peptides against rough LPS.

Lipopolysaccharide (LPS) is a bacterial endotoxin that can set off an immune cascade leading to septic shock and death. Some work has found evidence that (usually antimicrobial) peptides can (1) bind to LPS, (2) decrease cytokine production from macrophages, and (3) save mice from LPS-induced sepsis. Although it is not always the case that LPS binding is directly associated with rescuing mice. 

Here we attempt to design peptide binders for LPS. We do this by scoring binding of peptides against the lipid A core of LPS, and using score to drive a Bayesian optimization loop.

Current usage of this repo is to run umbrella sampling of peptides in serial through the use of the bash_scripts/run_process_sequences.sh script where you have an input file containing peptide IDs, sequence, and a short description (not used at the moment); an example file is included under input_files/. Separately, one could use the bayesopt_driver.py script to run simple tests of EVO_BO, an Bayesian optimization variant where an evolutionary algorithm is used to optimize the acquisition function. 