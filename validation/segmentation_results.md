
# Heart validation results

## Summary 

Note all comparisons are manual ~ automated.

### Pooled validation results

Comparison type | Pearson r | Pearson p | MSE | RMSE 
--- | --- | --- | --- | ---
Stroke volume (Fig. 1) | -0.15242324016138 | 0.06439896018496177 | 0.15472965577291617 | 0.393356906349585
Diastole (Fig. 2) | -0.26247640672559275 | 0.0012694995398936728 | 0.20613620826638715 | 0.4540222552545053
Systole (Fig. 3) | -0.11311310521647441 | 0.1710587220664049 | 0.1794835712596912 | 0.4236550144394507
Stroke volume ~ Diastole (Fig. 4) | -0.2530352842152165 | 0.0019165872000117688 | 0.2096502945634869 | 0.4578758506008882
Stroke volume ~ Systole (Fig. 5) | -0.13438963246682412 | 0.10342865304750616 | 0.1856839744406183 | 0.43091063393773227

### Frame by frame results
Comparison type | Pearson r | Pearson p | MSE | RMSE 
--- | --- | --- | --- | ---
15C, 25ppt, young (Fig. 6) | 0.687348518478324 | 2.846540382348946e-15 | 0.08067795437307526 | 0.284038649435381
20c, 15ppt, medium (Fig. 7) | 0.6013406437943658 | 3.870619630688776e-06 | 0.07848708943381695 | 0.2801554736816987

## Comparison plots

### Stroke volume

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/svm_sva.png'>

**Figure 1.** Validation results for manual and automated stroke volume measures.

### Diastole

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/mda_da.png'>

**Figure 2.** Validation results for manual and automated diastole measures.

### Systole

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/msa_sa.png'>

**Figure 3.** Validation results for manual and automated systole measures.

## Comparison of automated stroke volume with individual manual cardiac measures

### Stroke volume ~ diastole

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/svm_da.png'>

**Figure 4.** Validation results for manual stroke volume and automated diastole measures.

### Stroke volume ~ systole

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/svm_sa.png'>

**Figure 5.** Validation results for manual stroke volume and automated systole measures.

## Frame by frame comparisons

### Young (Temp: 15, Salinity: 25ppt, Timepoint: 1)

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/fr_15_25ppt_young_1.png'>

**Figure 6.** Validation results for frame by frame heart area measures.

### Medium (Temp: 20, Salinity: 25ppt, Timepoint: 1)

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/fr_20_15ppt_medium_1.png'>

**Figure 7.** Validation results for frame by frame heart area measures.


