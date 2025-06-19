# EvolveGen: Evolutional Data Generation Platform

🎯 *EvolveGen automatically creates high-quality synthetic datasets by **leveraging edge-side privacy-protected data to select and filter** cloud-generated synthetic datasets through a **evolutional loop** with offline or online **APIs**.*

👏 You can access **free online APIs** for text, image, video, audio, and more at [SiliconFlow](https://cloud.siliconflow.cn/models) or explore affordable options at [getimg.ai](https://dashboard.getimg.ai/models).
👏 You can implement a custom synthetic data selector by simply adding a new file to the `algo\client\selector` directory.

![Running Out of Data](https://media.nature.com/lw767/magazine-assets/d41586-024-03990-2/d41586-024-03990-2_50306276.jpg?as=webp)

AI's growth has relied on scaling neural networks and training on massive datasets, enabling LLMs like ChatGPT to handle conversations and reasoning (see [this nature paper](https://www.nature.com/articles/d41586-023-00641-w)). However, experts warn of limits as energy demands rise and training data dwindles (see [this nature paper](https://www.nature.com/articles/d41586-024-03408-z)). Epoch AI predicts AI training data will be exhausted by 2028 (see [this nature paper](https://www.nature.com/articles/d41586-024-01760-8)).

## Features  

- Enabling self-correction in synthetic dataset generation through evolution with refined prompts or privacy-protected data.
- User-friendly code style. 
- Customizable selector design for synthetic data quality evaluation, selection, and filtering.  
- Automate synthetic dataset generation with large generative model APIs across various modalities.  
- Automatically filter and iteratively refine high-quality synthetic datasets.  
- Flexible interfaces for seamless extension and customization. 

## How to Use  

1. **Prepare the Required Large Model APIs (Skip If Using Online APIs).**  
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
     - Iteratively generate, select, and provide feedback with privacy protection: `--framework Feedback`  
   - **Available Data Selectors (when `--framework Filter or Feedback`):**
     - Contrastive filter in [Private Contrastive Evolution (PCEvolve, ICML'25)](https://arxiv.org/abs/2506.05407): `--selector PCEvolve`
     - Similarity voting in [Private Evolution (PE, ICLR'24)](https://openreview.net/forum?id=YEhQs8POIo): `--selector PE`
     - Filtering using real data in [Real Filter (RF, ICLR'23)](https://openreview.net/forum?id=nUmCcZ5RKF): `--selector RF`

## Example

For a COVID-19 pneumonia detection task, generate 100 synthetic images per class based on 10 real and **private chest radiography (X-ray) images** on the edge using the Stable Diffusion API. The edge device utilizes a ResNet-18, with Private Evolution (PE) for selection and feedback provided with privacy protection:
```bash  
python -u main.py \
  -tt syn \        # Task Type: Only using the synthetic dataset for downstream task
  -tm I2I \        # Task Mode: Image to Image
  -f Feedback \    # Framework: Using the feedback mechanism
  -did 1 \         # GPU device ID
  -eps 0.2 \       # Privacy cost epsilon per iteration
  -rvpl 1 \        # Real and private volume per label
  -vpl 2 \         # Generated volume per label
  -oa 1 \          # Use online API
  -sgen StableDiffusionXL \  # Select StableDiffusionXL as the generative model
  -cret 1 \        # Other hyperparameter
  -cue ResNet18 \  # Edge client embedding model
  -cmodel ResNet18 \  # Edge client model
  -cmp 1 \         # Other hyperparameter
  -cef 1 \         # Other hyperparameter
  -cdata COVIDx \  # Private dataset
  -s PE            # Synthetic data selector: Private Evolution
```  