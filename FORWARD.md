# Forward

4B pretrain train step. The network field is 256³ at 4 nm. Each row is the native spacing, ordered z × y × x. Read sizes are the voxel counts that cover that field: `round(256 × 4 / nm)`.

Augment on every row is missing sections (probability 0.3, at most 2, fill zero), jitter (probability 0.5, ±2 voxels), noise (probability 0.3, std 0.01), and contrast (probability 0.5, gamma 0.8–1.25). On a 4 nm crop these run after the z downsample. On every coarser crop they run on the native crop.

| z × y × x | Read | Degrade | Augment | Upsample | Forward | L1 | Flow |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 × 4 × 4 nm | 256 × 256 × 256 | Nearest z to round(256 / zf), zf 2–10, depth 26–128. xy stays 256. | Missing, jitter ±2, noise, and contrast on that coarse-z stack. | Nearest image back to 256³. Target is already 256³. | Input and raw head are 256³. | Direct on 256³. Pool skipped. | Both endpoints 256³. Clean is the original volume. |
| 8 × 8 × 8 nm | 128 × 128 × 128 | Skipped. | Same defects on the 128³ crop. | Nearest ×2 of image and target to 256³. | Input and raw head are 256³. | Average-pool to 128³. Each window is the ×2 block. | Both endpoints 256³. Clean is the nearest repeat. |
| 16 × 16 × 16 nm | 64 × 64 × 64 | Skipped. | Same defects on the 64³ crop. | Nearest ×4 of image and target to 256³. | Input and raw head are 256³. | Average-pool to 64³. Each window is the ×4 block. | Both endpoints 256³. Clean is the nearest repeat. |
| 40 × 4 × 4 nm | 26 × 256 × 256 | Skipped. | Same defects on the 26 × 256 × 256 crop. | Nearest in z only, 26 to 256. xy already 256. | Input and raw head are 256³. | Average-pool to 26 × 256 × 256. Z windows overlap. xy unchanged. | Both endpoints 256³. Clean is the nearest repeat. |
| 30 × 6 × 6 nm | 34 × 171 × 171 | Skipped. | Same defects on the 34 × 171 × 171 crop. | Nearest on all axes to 256³. | Input and raw head are 256³. | Average-pool to 34 × 171 × 171. Z and xy windows overlap. | Both endpoints 256³. Clean is the nearest repeat. |
| 30 × 8 × 8 nm | 34 × 128 × 128 | Skipped. | Same defects on the 34 × 128 × 128 crop. | Nearest z, 34 to 256, and xy ×2, 128 to 256. | Input and raw head are 256³. | Average-pool to 34 × 128 × 128. Z windows overlap. xy is an exact ×2. | Both endpoints 256³. Clean is the nearest repeat. |

On train, a random zyx permutation is applied after the degraded image is on the 4 nm cube. The clean target is permuted with it, then nearest-upsampled, and the L1 uses that permuted size. Batch size stays 2. Validation skips the permutation. TensorBoard shows the 256³ cube on every row.
