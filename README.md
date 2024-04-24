# It Doesn’t Look Like Anything to Me: Using Diffusion Model to Subvert Visual Phishing Detectors

The repository contains major code and dataset to generate adversarial logo images that subvert Visual Phishing Detectors using the Diffusion model as the generator. Specifically, the adversarial diffusion model, LogoMorph, generates adversarial logo images that have a large perceptual distance from original logo images measured by visual phishing detectors but with semantic-preserving changes. We show visual examples of adversarial logo images and webpage screenshots that bypass the state-of-the-art visual phishing detectors. We also conduct a user study to evaluate the quality of our adversarial logos and webpage screenshots. 


## Usage
We train the unconditional diffusion for each target phishing brand: 

### Train LogoMorph
Using Bank of America (BOA) as an example, the example command to train the adversarial diffusion model image generator is given in examples/diffusion/train.sh. 

### Sample Logo Images
The example command to sample adversarial logo images from pretrained model weights is given in examples/diffusion/sample.sh. You can either train your own model or use our pretrained weights. Our pretrained weights will be uploaded later.  

