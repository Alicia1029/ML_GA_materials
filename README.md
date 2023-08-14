# ml-ga-invert

Repository for Master project: inverting latent space.

Sections include different codes used to output the results of the project.

### Section 3.1 

3.1 has checkpoint and outputs, example structures that trained and generated by CDVAE model.
gen_structure.py file extracts structures from the eval_gen.pt file
generate-ls.ipynb is the notebook that includes t-SNE plot to visualise latent space

### Section 3.2

Ag_cif contains five reference Ag structures from the Materials Project
3.2.2 is the notebook that genetic algorithm without using relaxation function
3.2.3 is GA with CHGNet in relaxation function, structures encoded by CDVAE
3.2_test file tests the cosine similarity and energies of Ag structures encoded by CDVAE

### Section 3.3 shows the invariance problem of CDVAE

### Section 3.4

only-chgnet is GA with CHGNet in relaxation function, structures also encoded by CHGNet
chgnet-test is the notebook that tests the cosine similarity of Ag structures encoded by CDVAE


