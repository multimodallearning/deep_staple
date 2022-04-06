# 20220319 - crossmoda_convex_adam_lr.pth

Registered data from CrossModa challenge with convex adam
crossmoda_"target" -> crossmoda_"source"
                T2 -> T1
            moving -> fixed

900 registrations (1F<-30M)x30 (right<-right) + 900 registrations (1F<-30M)x30  (left<-left) = 1800 registrations

# load with:
torch.load()

contains nested dicts with dict[fixed_id][moving_id] with sparse label tensors.
Use .to_dense() to unpack

# Steps to reproduce:

Use run_registration_tumour_t1_t2_crossmoda.ipynb