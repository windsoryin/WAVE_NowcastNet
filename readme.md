# DRCAT

:triangular_flag_on_post:(Nov 13, 2025): This paper is currently under submission, therefore this repo is currently under preparation. 

This the origin implementation of WAVE-NowcastNet in the following paper:
<WAVE-NowcastNet: A Novel Deep Learning Framework for Enhancing Precipitation Nowcasting with GNSS-Derived Precipitable Water Vapor>

This repository provide example code to run model visualizations and source data.
<p align="center">
<img src=".\pic\Network.png" height = "360" alt="" align=center />
<br><br>
<b>Figure 1.</b> The architecture of WAVE-NowcastNet.
</p>


## Usage

run Figure3-5.py can generate figures used in this study using inference data predicted by different models.
Due to the size limit of Github, only two events files are uploaded in this repo. For Full inference file, you can download in this googledrive link:

for example:

```bash
# figure 3
python Figure3_prediction_comparison.py

```

<p align="center">
<img src="./comparison_plots/comparison_event_039.png" height = "360" alt="" align=center />
<br><br>
<b>Figure 2.</b> Prediction results.
</p>


## Contact
For feedback and questions, please contact us at windosryin@whu.edu.cn