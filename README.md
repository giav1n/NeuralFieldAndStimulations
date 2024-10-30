# NeuralFieldAndStimulations
Using NeuralFields.jl we build a model of moouse neural cortex (25 $mm^2$) unbder anesthesia. In this condition we observe spontaneous traveling wawes originating at the corners of the field.
We also implemented a stimulation protocol where we target one random cortical module at the center of the field each second of simulation. In some instances this pertubation causes spiral or planar wawes that are commonly observed in real data.

![Description of GIF](Animation.gif)

In the folder SNN-Simulations you can find the code, based on NEST, for the spiking neural network simulation that uses the same connectivity and single cortical module parameters.
