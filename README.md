# audio-source-separation

## Quickstart


## Objectives (according to guidelines)
- [ ] Introduce the problem of source separation (in the report)
- [ ] Conduct a comprehensive analysis of the given dataset
- [ ] Implement two methods to solve the problem (Spectro_Unet + Wave_Unet)
- [ ] Compare the results with evaluation metrics (quantitative + qualitative) compared to baseline
- [ ] Discuss / implement strategies to optimize (data, model)

## Todo

### Data
- [x] create the dataloader
- [ ] analyze data (high level stats, visualization, etc.)
- [ ] use torchaudio transform from the preprocessing pipeline: https://docs.pytorch.org/audio/stable/transforms.html
- [ ] filter data by snr levels (to have models that work better)

### Models
- [ ] implement two approaches for audio-source separation
    - [ ] On based on the spectrogram: U-Net: A. Jansson et Al., SINGING VOICE SEPARATION WITH DEEP U-NET CONVOLUTIONAL NETWORK, ISMIR 2017
    - [ ] On based on the wave directly: Wave U-Net: A multi-scale neural network for end-to-end audio source separation, ISMIR 2018

### Misc
- [ ] implement evaluation metrics

### Optimisation
- [ ] run hyperparameter optimizations
- [ ] rethink architecture (ex. using separable convolutions)
- [ ] implement data augmentation with a Transform passed to the dataloader