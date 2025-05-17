# Noncyclic Scheduling of Cluster Tools Using Deep Reinforcement Learning

This repository contains experiments on noncyclic scheduling of cluster tools using Deep Reinforcement Learning. Each folder includes DRL models, conventional robot move sequence algorithms, experimental results in dual/single-armed cluster tools.

## Repository Structure

- **dual_arm_cluster_tool**: 
  - **concurrent_flow**: Experiments and models for concurrent flow in dual-armed cluster tools.
  - **mlif_flow**: Experiments and models for Multi-Lot-per-FOUP Flow.
  - **purge**: Experiments and models for Purge constraint.
  - **skip_flow**: Experiments and models for Skip Flow.

- **single_arm_cluster_tool**:
  - **concurrent_flow**: Experiments and models for concurrent flow in single-armed cluster tools.
  - **mlif_flow**: Experiments and models for Multi-Lot-per-FOUP Flow.
  - **purge**: Experiments and models for Purge constraint.
  - **skip_flow**: Experiments and models for Skip Flow.

Each folder contains a `notebook.ipynb` file, which provides detailed experimental results for each instance.

## Example Results

Below is an example based on the experimental results from `dual_arm_cluster_tool/concurrent_flow/notebook.ipynb`. This example shows the average and instance-specific results for both baseline and DRL outcomes.

### Average Results

- **CSS Makespan Average**: 12770.30
- **RL Makespan Average**: 11545.11

### Instance-Specific Results Example

| InstanceID | Type 1 Process Time     | Type 2 Process Time     | CSS Makespan | RL Makespan |
|------------|-------------------------|-------------------------|--------------|-------------|
| 0          | [229, 60, 211, 0, 0, 0] | [0, 0, 0, 76, 123, 133] | 11843        | 9380        |
| 1          | [197, 126, 244, 0, 0, 0] | [0, 0, 0, 112, 21, 284] | 14628        | 13783       |
| 2          | [65, 139, 92, 0, 0, 0]   | [0, 0, 0, 159, 275, 290]| 9883         | 9927        |
| 3          | [128, 192, 258, 0, 0, 0] | [0, 0, 0, 153, 251, 178]| 13183        | 13165       |
| 4          | [126, 159, 22, 0, 0, 0]  | [0, 0, 0, 187, 182, 159]| 9802         | 9745        |
| 5          | [125, 133, 164, 0, 0, 0] | [0, 0, 0, 60, 258, 218] | 11542        | 10769       |
| 6          | [162, 34, 32, 0, 0, 0]   | [0, 0, 0, 60, 48, 78]   | 8607         | 6448        |
| 7          | [117, 36, 91, 0, 0, 0]   | [0, 0, 0, 125, 120, 148]| 7859         | 7763        |
| 8          | [281, 277, 234, 0, 0, 0] | [0, 0, 0, 11, 26, 244]  | 14329        | 13669       |

Detailed experimental results for each case can be found in the corresponding notebook.ipynb files within each experiment folder.

## Model and Training Parameters

- **Model Parameters**:
  - Embedding dimension \(d\): 256
  - Number of encoder layers: 3
  - Number of attention heads: 16
  - Feed-forward hidden dimension: 512
  - Key, value, and query dimensions: 16

- **Training Parameters**:
  - Number of epochs \(E\): 100
  - Steps per epoch: 10000
  - Batch size \(B\): 64
  - Optimizer: Adam
  - Learning rate: 0.0001

- **Hardware**:
  - CPU: i9-10980
  - RAM: 64GB
  - GPU: RTX 3090

## References

This work is inspired by and builds upon the following research and resources:

- Lee, J. H., Kim, H. J., & Lee, T. E. (2014). Scheduling cluster tools for concurrent processing of two wafer types. IEEE Transactions on Automation Science and Engineering, 11(2), 525-536.
- Lee, J. H., Kim, H. J., & Lee, T. E. (2015). Scheduling cluster tools for concurrent processing of two wafer types with PM sharing. International Journal of Production Research, 53(19), 6007-6022.
- Yu, T. S., Kim, H. J., & Lee, T. E. (2017). Scheduling single-armed cluster tools with chamber cleaning operations. IEEE Transactions on Automation Science and Engineering, 15(2), 705-716.
- Kim, H., Kim, H. J., Lee, J. H., & Lee, T. E. (2013). Scheduling dual-armed cluster tools with cleaning processes. International Journal of Production Research, 51(12), 3671-3687.

Additionally, the code and methodologies are influenced by the POMO framework, which can be found at [POMO GitHub Repository](https://github.com/yd-kwon/POMO).
