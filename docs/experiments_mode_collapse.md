# Experiments with cgan architecture 2018-08-30

- Creating a down path for reaction components and then upsampling a concatinated results worked
- Unet did not perform well
- Mode collapse an issue
- Added noise to compressed representation of reaction

To tackle mode collapse

|Test| Comments | Blast| Unique| Looks same after | Path|
|-----|-----|-----|-----|-----|-----|
|Baseline | Sequences looks great. M letter at the begginnig and stable | 1,200 | 2 | 2,692 | hinge_loss/batch_size=64/d_dim_8_g_dim_8/d_lr_0.0002_g_lr_0.0002_b1_0.0_b2_0.9_k_3x3_d_2_subpixel_n_6 |
|Added std to the loss of the gen | Sequences looks ok. M letter at the begginnig and stable. Variation loss does not decrease significantly | 3,600 | 56 | 2,991 | hinge_loss_std/batch_size=64/d_dim_8_g_dim_8/d_lr_0.0002_g_lr_0.0002_b1_0.0_b2_0.9_k_3x3_d_2_subpixel_n_6  |
|Added 5x std to the loss of the gen | Sequences looks ok. M letter at the begginnig and stable. Variation loss does not decrease significantly| Did not happen | 31 | 2,991 |hinge_loss_std_x5/batch_size=64/d_dim_8_g_dim_8/d_lr_0.0002_g_lr_0.0002_b1_0.0_b2_0.9_k_3x3_d_2_subpixel_n_6|
|Added 10x std to the loss of the gen | Sequences looks ok. M letter at the begginnig and stable. Variation loss does not decrease| 4,801 | 27 | 2,692|hinge_loss_std_x5/batch_size=64/d_dim_8_g_dim_8/d_lr_0.0002_g_lr_0.0002_b1_0.0_b2_0.9_k_3x3_d_2_subpixel_n_6|
|Added 50x std to the loss of the gen | M is not stable. Sequences are not the same, however, it is not the variation we expect | 3601 | 62 at the step 4K | Does not get the same, however, very similar |hinge_loss_std_x50/batch_size=64/d_dim_8_g_dim_8/d_lr_0.0002_g_lr_0.0002_b1_0.0_b2_0.9_k_3x3_d_2_subpixel_n_6|
|Added mini batch discrimination | M is not stable. Sequences are not the same (better than with std), however, it is not the variation we expect |  2,401 | 64 | Same variation even after 10000 steps | hinge_loss_mini_batch/batch_size=64/d_dim_8_g_dim_8/d_lr_0.0002_g_lr_0.0002_b1_0.0_b2_0.9_k_3x3_d_2_subpixel_n_6 |
|Added mini batch discrimination and double the noise | Eventually it gets the same |   6,001 | 48 | 7,177 | hinge_loss_mini_batch_noise_1/batch_size=64/d_dim_8_g_dim_8/d_lr_0.0002_g_lr_0.0002_b1_0.0_b2_0.9_k_3x3_d_2_subpixel_n_6 |
|Added mini batch discrimination and half the size of hidden layers| M is not stable. Sequences are not the same, however, it is not the variation we expect | Never | 64| Same variation even after 10000 steps | hinge_loss_mini_batch/batch_size=64/d_dim_4_g_dim_4/d_lr_0.0002_g_lr_0.0002_b1_0.0_b2_0.9_k_3x3_d_2_subpixel_n_6 |
|Added mini batch discrimination and double the size of hidden layers| M is stable. Sequences are not the same, however, the variation very small |  2,401 | 56 | Tiny variation even after 10000 steps | hinge_loss_mini_batch/batch_size=64/d_dim_16_g_dim_16/d_lr_0.0002_g_lr_0.0002_b1_0.0_b2_0.9_k_3x3_d_2_subpixel_n_6 |
|Added mini batch discrimination and double generator steps | M is stable. Sequences are not the same, however, the variation very small |  3,601 | 64 |Very tiny variation even after 10000 steps | hinge_loss_mini_batch/batch_size=64/d_dim_16_g_dim_16/d_lr_0.0002_g_lr_0.0002_b1_0.0_b2_0.9_k_3x3_d_2_subpixel_n_6 |
|Added mini batch discrimination and static smiles embeddings | M is not stable. Sequences are not the same, however, the variation very small |  3,601 | 64 | Tiny variation even after 10000 steps | hinge_loss_mini_batch_smiles_vec/batch_size=64/d_dim_8_g_dim_8/d_lr_0.0002_g_lr_0.0002_b1_0.0_b2_0.9_k_3x3_d_2_subpixel_n_6 |
|Added mini batch discrimination and constant number of hidden layers | M is stable. Sequences are not the same, however, the variation very small |  3,601 | 64 | Tiny variation even after 10000 steps | hinge_loss_mini_batch/batch_size=64/d_dim_64_g_dim_64/d_lr_0.0002_g_lr_0.0002_b1_0.0_b2_0.9_k_3x3_d_2_subpixel_n_6 |
|Added mini batch discrimination and conditional batch norm | M is not stable. Sequences are not the same | 4,801 | 64 | Same variation even after 10000 steps | hinge_loss_mini_batch_cb/batch_size=64/d_dim_8_g_dim_8/d_lr_0.0002_g_lr_0.0002_b1_0.0_b2_0.9_k_3x3_d_2_subpixel_n_6 |
| Net v2 | M is stable. Sequences are not the same. Blast results are constantly ok | 1,201 | 64 | Same variation even after 10000 steps | 8x128_embedding/hinge_loss_mini_batch/batch_size=64/d_dim_64_g_dim_64/d_lr_0.0002_g_lr_0.0002_b1_0.0_b2_0.9_k_3x3_d_2_subpixel_n_6 |
