# CoAutoGen: Cloud-Edge Collaboration Platform for Automated Synthetic Dataset Generation

*CoAutoGen tackles dwindling training data by integrating dataset generation methods with cloud-edge collaboration, enabling **automatic scoring and evolution** in synthetic dataset creation via a **feedback loop** using refined prompts, privacy-protected data, and local or online **APIs**.*

![Running Out of Data](https://media.nature.com/lw767/magazine-assets/d41586-024-03990-2/d41586-024-03990-2_50306276.jpg?as=webp)


AI's growth has relied on scaling neural networks and training on massive datasets, enabling LLMs like ChatGPT to handle conversations and reasoning (see [this nature paper](https://www.nature.com/articles/d41586-023-00641-w)). However, experts warn of limits as energy demands rise and training data dwindles (see [this nature paper](https://www.nature.com/articles/d41586-024-03408-z)). Epoch AI predicts AI training data will be exhausted by 2028 (see [this nature paper](https://www.nature.com/articles/d41586-024-01760-8)).

## Features  

- Enabling self-correction in synthetic dataset creation through feedback with refined prompts or privacy-protected data.
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
   - **Available Raters (when `--framework Filter or Feedback`):**  
     - Histogram rating in [Private Evolution (PE, ICLR'24)](https://openreview.net/forum?id=YEhQs8POIo): `--rater PE`  
     - Real data rating in [Real Filter (RF, ICLR'23)](https://openreview.net/forum?id=nUmCcZ5RKF): `--rater RF`  

## Example

For a COVID-19 pneumonia detection task, generate 100 synthetic images per class based on 10 real and **private chest radiography (X-ray) images** on the edge using the Stable Diffusion API. The edge device utilizes a ResNet-18, with Private Evolution (PE) for rating and feedback provided with privacy protection:
```bash  
python -u main.py \
  -tt syn \       # Task Type: Only using the synthetic dataset for downstream task
  -tm I2I \       # Task Mode: Image to Image
  -f Feedback \   # Framework: Feedback mechanism
  -did 1 \        # GPU device ID
  -eps 5 \        # Privacy budget epsilon
  -rvpl 10 \      # Real volume per label
  -vpl 100 \      # Generated volume per label
  -sgen StableDiffusion \  # Stable Diffusion API
  -cret 1 \       # Other hyperparameter
  -cue ResNet18 \ # Edge client embedding model
  -cmodel ResNet18 \ # Edge client model
  -cmp 1 \        # Other hyperparameter
  -cef 1 \        # Other hyperparameter
  -cdata COVIDx \ # Pravate dataset
  -r PE           # Rater: Private Evolution
```  