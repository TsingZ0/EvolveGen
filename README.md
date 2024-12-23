# EdgeAutoSyn: Cloud-Edge Collaboration Platform for Automated Synthetic Dataset Generation

AI's growth has relied on scaling neural networks and training on massive datasets, enabling LLMs like ChatGPT to handle conversations and reasoning (see [this nature paper](https://www.nature.com/articles/d41586-023-00641-w)). However, experts warn of limits as energy demands rise and training data dwindles (see [this nature paper](https://www.nature.com/articles/d41586-024-03408-z)). Epoch AI predicts AI training data will be exhausted by 2028 (see [this nature paper](https://www.nature.com/articles/d41586-024-01760-8)).

While internet content grows under 10% annually, AI datasets double each year. Content providers increasingly block web crawlers or restrict data use (see [this paper](https://arxiv.org/abs/2407.14933)). By 2024, blocking tokens on high-quality content rose from 3% in 2023 to 20-33% (see [this nature paper](https://www.nature.com/articles/d41586-024-03990-2)).

## Features  

- Create "dataset twins" for private datasets with differential privacy guarantees.  
- Intuitive, user-friendly code style.  
- Customizable rater design for synthetic data rating, selection, and filtering.  
- Automate synthetic dataset generation with large generative model APIs across various modalities.  
- Automatically filter and iteratively refine high-quality synthetic datasets.  
- Flexible interfaces for seamless extension and customization.  

## How to Use  

1. **Prepare Required Large Model APIs**  
   - Set up large model APIs either through local deployment (downloading model weights for captioner, generator, LLM, etc.) or via online-accessible APIs.  

2. **Set Up the Environment**  
   - Install [CUDA](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).  
   - Install the [latest Conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) and activate it.  
   - Create the Conda environment:  
     ```bash  
     conda env create -f env_cuda_latest.yaml  
     # You may need to downgrade PyTorch using pip to match the CUDA version  
     ```  

3. **Run the Script**  
   - Execute `python main.py` with your configurations.  
   - **Available Frameworks:**  
     - Generate with prompts: `--framework Gen`  
     - Generate with LLM-enhanced prompts: `--framework GenLLM`  
     - Iteratively generate, filter, and accumulate: `--framework Filter`  
     - Iteratively generate, rate, and provide feedback with privacy protection: `--framework Feedback`  
   - **Available Raters:**  
     - Histogram rating in [Private Evolution (PE, ICLR'24)](https://openreview.net/forum?id=YEhQs8POIo): `--rater PE`  
     - Real data rating in [Real Filter (RF, ICLR'23)](https://openreview.net/forum?id=nUmCcZ5RKF): `--rater RF`  