# Datasets

## Sources

- BraTS2021: <https://arxiv.org/abs/2107.02314>
- LUMIERE: <https://springernature.figshare.com/collections/The_LUMIERE_Dataset_Longitudinal_Glioblastoma_MRI_with_Expert_RANO_Evaluation/5904905/1> (paper: <https://www.nature.com/articles/s41597-022-01881-7>)
- UPENN GBM: <https://www.nature.com/articles/s41597-022-01560-7#code-availability>
- Rembrandt: <https://pubmed.ncbi.nlm.nih.gov/30106394/>
- UCSF-PDGM: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9748624/>
- EGD: <https://www.sciencedirect.com/science/article/pii/S2352340921004753>
- TCGA: <https://www.nature.com/articles/sdata2017117>

## Data preprocessing

1. Register all images to the SRI24 (Rohlfing et al., 2009)
2. Skull-strip all images using HD-BET (Isensee et al., 2019)
3. Segment tumors with automatically with your favorite pre-trained brain tumor segmentation model.
4. Re-arrange data into BIDS structure (<https://bids.neuroimaging.io/>)

```
.
├── dataset_001
│   ├── sub-001
│   │   ├── preop
│   │   │   ├── sub-001_ses-preop_space-sri_flair.nii.gz
│   │   │   ├── sub-001_ses-preop_space-sri_t1.nii.gz
│   │   │   ├── sub-001_ses-preop_space-sri_t1c.nii.gz
│   │   │   ├── sub-001_ses-preop_space-sri_t2.nii.gz
│   │   │   ├── sub-001_ses-preop_space-sri_seg.nii.gz
...
```

5. Run `uv run scripts/create_glioma_splits.py` to create the train, val and test splits based on the patient IDs in `glioma_splits/{train,val,test}.txt`.
   The script create a new structure based on symbolic links.
6. Update dataset paths in config files in `dinov2/configs/` accordingly. Options that need to be updated: train.dataset_path, evaluation.train_dataset_path, evaluation.val_dataset_path, evaluation.test_dataset_path.
