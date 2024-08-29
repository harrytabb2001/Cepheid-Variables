# Cepheid Variables 
## Summary
Fifth lab experiment from my second year undergraduate in Physics at The University of Manchester.

The aim of this experiment was to calculate the distance to spiral galaxy NGC 4258 by measuring the intensity variation
of 15 variable stars (Cepheid Variables) in the outer regions of the galaxy. Light curves were constructed by writing a phase
folding algorithm and an experimentally determined relationship between the period of intensity
variation and absolute magnitude was used to calculate the distance to each star. From this distance, the recession velocity of the galaxy was used to calculate a value for the Hubble constant.  It is worth noting that instead of using a niche astrophysics software (DS9) I opted to replicate the software in Python to test my coding abilities. Ultimately, this replicated the same results but was more versatile for this experiment.

## Technical Highlights
* Principal Component Analysis
* Writing a user friendly program in Python to replace DS9 software
* Data visualisation using Matplotlib
* Dataset expansion methods (phase folding)
* Fourier analysis
* Chi-Square analysis

## Files
* Cepheid main.py contains all of the data analysis code used in the experiment. It is worth noting that I went significantly beyond what was required. The lab script recommended
using ds9 software; however, as a challenge, I wrote my own Python script to carry out the experiment and perform all data analysis. This was a valuable learning experience.
* Cepheid Variables Lab Script.pdf contains a brief outline of the experiment given to me before I started.
* Cepheid Lab Notebook.pdf contains my handwritten notes from during the experiment including a schematic diagram, key figures, snippets of the recorded data, calculations, data analysis and conclusions.
### Note that due to the amount of images in this file, they struggle to load. To view in full, please download. 
* Cepheid Variables Lab Report.pdf contains my written up lab report from after the experiment.
