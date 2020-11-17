
# Heart validation results (mean px)

## Summary 

Note all comparisons are manual ~ automated.

### Paleomon 

Comparison type | Pearson r | Pearson p | MSE | RMSE | Offset mean | Offset sd
--- | --- | --- | --- | --- | --- | ---
Raw area (Fig. 1) | 0.5764570017085253 | 4.62815792866711e-37 | 0.07365128547328822 | 0.271387703246275 |  |  
Stroke volume (Fig. 2) | -0.0822090918909174 | 0.7621380849027015 | 0.20925523428721365 | 0.4574442417248398 |  | 
Diastole | -0.11753274460880733 | 0.6646494499578716 | 0.30297905367577677 | 0.5504353310569524 | 2.0 | 3.9210967853395307
Systole | 1.0 | 0.0 | 0.39651998916230224 | 0.6296983318719387 | 0.3125 | 0.9164298936634487

### Radix 

Comparison type | Pearson r | Pearson p | MSE | RMSE | Offset mean | Offset sd
--- | --- | --- | --- | --- | --- | ---
Raw area (Fig. 3) | 0.4833411415509912 | 4.187909238022705e-13 | 0.0801680477442436 | 0.28313962588137254 |  | 
Stroke volume (Fig. 4) | 0.01586277522478377 | 0.9400096127113523 | 0.16886811635905094 | 0.410935659634268 |  | 
Diastole | -0.04264201025548669 | 0.8396148566045388 | 0.16243691633042454 | 0.4030346341574438 | 0.92 | 0.7959899496852959
Systole | 0.9999999999999999 | 1.5864253030664577e-181 | 0.17138693279375186 | 0.4139890491229833 | 0.24 | 0.9911609354691094

## Comparison plots - Paleomon

### Raw area 

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/paleomon_areas.png'>

**Figure 1.** Validation results for manual and automated raw area measures.

### Stroke volume

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/paleomon_sv.png'>

**Figure 2.** Validation results for manual and automated stroke volume measures.

### Diastole

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/paleomon_d.png'>

**Figure 3.** Validation results for manual and automated diastole measures.

### Systole

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/paleomon_s.png'>

**Figure 4.** Validation results for manual and automated systole measures.

## Comparison plots - radix

### Raw area 

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/radix_areas.png'>

**Figure 5.** Validation results for manual and automated raw area measures.

### Stroke volume

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/radix_sv.png'>

**Figure 6.** Validation results for manual and automated stroke volume measures.

### Diastole

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/radix_d.png'>

**Figure 7.** Validation results for manual and automated diastole measures.

### Systole

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/radix_s.png'>

**Figure 8.** Validation results for manual and automated systole measures.

    