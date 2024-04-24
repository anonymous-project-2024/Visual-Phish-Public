# It Doesn’t Look Like Anything to Me: Using Diffusion Model to Subvert Visual Phishing Detectors

The repository contains major code and dataset to generate adversarial logo images that subvert Visual Phishing Detectors using the Diffusion model as the generator. Specifically, the adversarial diffusion model, LogoMorph, generates adversarial logo images that have a large perceptual distance from original logo images measured by visual phishing detectors but with semantic-preserving changes. We show visual examples of adversarial logo images and webpage screenshots that bypass the state-of-the-art visual phishing detectors. We also conduct a user study to evaluate the quality of our adversarial logos and webpage screenshots. 


## Examples

### Train LogoMorph
The example to train the adversarial diffusion model image generator is given in train.sh. 

### Sample Logo Images
The example command to sample adversarial logo images from pretrained model weights is given in sample.sh. The model weights will be uploaded soon. 

