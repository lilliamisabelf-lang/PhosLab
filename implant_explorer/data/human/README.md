# Human dataset

Most of `data/human/` is gitignored because it would balloon the repo to >10 GB
(full HCP subjects, FreeSurfer recons, neuropythy built-ins).

What is shipped, and why:

- `demo_subject/subjects/fsaverage/T1w/fsaverage/mri/inferred_varea.mgz`
- `demo_subject/subjects/fsaverage/T1w/fsaverage/mri/inferred_eccen.mgz`
- `demo_subject/subjects/fsaverage/T1w/fsaverage/mri/inferred_angle.mgz`

These ~1.3 MB of inferred retinotopy maps are the minimum the explorer needs
for `--dataset human_demo --human-subject fsaverage` to load. Everything else
(individual subject anatomies, full FreeSurfer trees, neuropythy built-ins) is
generated locally and stays out of git.

To run other subjects (e.g. `100610`), follow the preprocessing docs under
`documentation/`.
