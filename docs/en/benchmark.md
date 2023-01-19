# Benchmark

We compare our results with some popular frameworks and official releases in terms of speed.

## Settings

### Software Environment

- Python 3.7
- PyTorch 1.10
- CUDA 11.1
- CUDNN 8.1.0

## Main Results

### SceneNeRF

#### NeRF

<table>
	<tr>
	    <th rowspan="2">test data</th>
        <th colspan="2">PSNR</th>
        <th colspan="2">SSIM</th>
	</tr >
	<tr>
	    <th>NeRF</th>
	    <th>XRNeRF</th>
	    <th>NeRF</th>
	    <th>XRNeRF</th>
	</tr >
	<tr >
	    <td>blender_chair</td>
        <td>33.927</td> <td>34.528</td> <td>0.967</td> <td>0.985</td>
	</tr>
	<tr >
	    <td>blender_drums</td>
        <td>25.600</td> <td>25.685</td> <td>0.925</td> <td>0.946</td>
	</tr>
	<tr >
	    <td>blender_ficus</td>
        <td>30.13</td> <td>29.300</td> <td>0.964</td> <td>0.972</td>
	</tr>
	<tr >
	    <td>blender_hotdog</td>
        <td>36.18</td> <td>	35.905</td> <td>0.974</td> <td>0.985</td>
	</tr>
	<tr >
	    <td>blender_materials</td>
        <td>29.62</td> <td>	29.014</td> <td>0.949</td> <td>0.967</td>
	</tr>
	<tr >
	    <td>blender_mic</td>
        <td>32.58</td> <td>32.95</td> <td>0.980</td> <td>0.986</td>
	</tr>
	<tr >
	    <td>blender_ship</td>
        <td>28.65</td> <td>29.46</td> <td>0.856</td> <td>0.932</td>
	</tr>
	<tr >
	    <td>llff_fern</td>
        <td>25.17</td> <td>26.277</td> <td>0.792</td> <td>0.892</td>
	</tr>
	<tr >
	    <td>llff_flower</td>
        <td>27.40</td> <td>26.592</td> <td>0.827</td> <td>0.884</td>
	</tr>
	<tr >
	    <td>llff_fortress</td>
        <td>31.16</td> <td>31.485</td> <td>0.881</td> <td>0.952</td>
	</tr>
	<tr >
	    <td>llff_horns</td>
        <td>27.45</td> <td>26.162</td> <td>0.828</td> <td>0.895</td>
	</tr>
	<tr >
	    <td>llff_leaves</td>
        <td>20.92</td> <td>19.749</td> <td>0.690</td> <td>0.668</td>
	</tr>

</table>


#### Kilo-NeRF

<table>
	<tr>
	    <th rowspan="2">test data</th>
        <th colspan="2">PSNR</th>
        <th colspan="2">SSIM</th>
        <th colspan="2">elapsed_time(ms)</th>
	</tr >
	<tr>
	    <th>KiloNeRF</th>
	    <th>XRNeRF</th>
	    <th>KiloNerf</th>
	    <th>XRNeRF</th>
	    <th>KiloNerf</th>
	    <th>XRNeRF</th>
	</tr >
	<tr >
	    <td>nsvf_Synthetic_NeRF_chair</td>
        <td>33.044</td> <td>33.037</td> <td>0.971</td> <td>0.979</td> <td>384.98</td> <td>407.78</td>
	</tr>
	<tr >
	    <td>nsvf_Synthetic_NeRF_drums</td>
        <td>25.327</td> <td>25.308</td> <td>0.931</td> <td>0.949</td> <td>413.03</td> <td>353.62</td>
	</tr>
	<tr >
	    <td>nsvf_Synthetic_NeRF_ficus</td>
        <td>30.1</td> <td>30.176</td> <td>0.967</td> <td>0.975</td> <td>351.04</td> <td>337.22</td>
	</tr>
	<tr >
	    <td>nsvf_Synthetic_NeRF_hotdog</td>
        <td>32.316</td> <td>33.408</td> <td>0.974</td> <td>0.986</td> <td>484.22</td> <td>491.49</td>
	</tr>
	<tr >
	    <td>nsvf_Synthetic_NeRF_lego</td>
        <td>33.398</td> <td>33.381</td> <td>0.971</td> <td>0.982</td> <td>379.1</td> <td>365.16</td>
	</tr>
	<tr >
	    <td>nsvf_Synthetic_NeRF_materials</td>
        <td>29.193</td> <td>29.175</td> <td>0.951</td> <td>0.966</td> <td>380.28</td> <td>358.57</td>
	</tr>
	<tr >
	    <td>nsvf_Synthetic_NeRF_mic</td>
        <td>33.186</td> <td>33.346</td> <td>0.982</td> <td>0.987</td> <td>370.31</td> <td>346.71</td>
	</tr>
	<tr >
	    <td>nsvf_Synthetic_NeRF_ship</td>
        <td>28.892</td> <td>29.295</td> <td>0.874</td> <td>0.933</td> <td>491.92</td> <td>488.35</td>
	</tr>
	<tr >
	    <td>Average</td>
        <td>30.68</td> <td>30.89102</td> <td>0.9526</td> <td>0.9697</td> <td>406.86</td> <td>393.61</td>
	</tr>

</table>

#### Mip-NeRF

<table>
	<tr>
	    <th rowspan="3">MultiScale Blender</th>
        <th align="center" colspan="8">PSNR</th>
	</tr >
	<tr>
	    <th align="center" colspan="2">800x800</th>
	    <th align="center" colspan="2">400x400</th>
	    <th align="center" colspan="2">200x200</th>
	    <th align="center" colspan="2">100x100</th>
	</tr >
	<tr>
	    <th>Jax</th>
	    <th>XRNeRF</th>
	    <th>Jax</th>
	    <th>XRNeRF</th>
	    <th>Jax</th>
	    <th>XRNeRF</th>
	    <th>Jax</th>
	    <th>XRNeRF</th>
	</tr >
	<tr >
	    <td>blender_ship</td>
        <td>29.599</td> <td>28.522</td> <td>31.955</td> <td>30.754</td> <td>33.845</td> <td>32.848</td> <td>34.868</td> <td>33.754</td>
	</tr>
	<tr >
	    <td>blender_mic</td>
        <td>33.739</td> <td>32.478</td> <td>36.353</td> <td>35.008</td> <td>38.837</td> <td>37.958</td> <td>39.011</td> <td>38.064</td>
	</tr>
	<tr >
	    <td>blender_materials</td>
        <td>30.128</td> <td>29.278</td> <td>31.424</td> <td>30.505</td> <td>33.163</td> <td>32.192</td> <td>34.174</td> <td>33.122</td>
	</tr>
	<tr >
	    <td>blender_lego</td>
        <td>33.971</td> <td>32.803</td> <td>35.248</td> <td>34.123</td> <td>35.796</td> <td>34.848</td> <td>35.223</td> <td>34.382</td>
	</tr>
	<tr >
	    <td>blender_hotdog</td>
        <td>36.457</td> <td>35.803</td> <td>38.382</td> <td>37.631</td> <td>39.831</td> <td>39.096</td> <td>39.935</td> <td>39.038</td>
	</tr>
	<tr >
	    <td>blender_ficus</td>
        <td>31.490</td> <td>29.222</td> <td>32.267</td> <td>30.093</td> <td>33.255</td> <td>31.655</td> <td>33.606</td> <td>31.785</td>
	</tr>
	<tr >
	    <td>blender_drums</td>
        <td>25.297</td> <td>24.790</td> <td>26.463</td> <td>26.020</td> <td>27.808</td> <td>27.510</td> <td>28.791</td> <td>28.369</td>
	</tr>
	<tr >
	    <td>blender_chair</td>
        <td>33.351</td> <td>32.429</td> <td>36.517</td> <td>35.618</td> <td>38.056</td> <td>37.342</td> <td>37.950</td> <td>37.257</td>
	</tr>
	<tr >
	    <td>Average</td>
        <td>31.754</td> <td>30.666</td> <td>33.576</td> <td>32.469</td> <td>35.074</td> <td>34.181</td> <td>35.445</td> <td>34.472</td>
	</tr>

</table>



#### InstantNGP

<table>
	<tr>
	    <th rowspan="2">test data</th>
        <th colspan="2">PSNR</th>
	</tr >
	<tr>
	    <th>InstantNGP</th>
	    <th>XRNeRF</th>
	</tr >
	<tr >
	    <td>blender_chair</td>
        <td>32.927</td> <td>32.71</td>
	</tr>
	<tr >
	    <td>blender_drums</td>
        <td>26.02</td> <td>26.9</td>
	</tr>
	<tr >
	    <td>blender_ficus</td>
        <td>33.51</td> <td>33.97</td>
	</tr>
	<tr >
	    <td>blender_hotdog</td>
        <td>37.40</td> <td>	37.17</td>
	</tr>
	<tr >
	    <td>blender_lego</td>
        <td>36.39</td> <td>35.1</td>
	</tr>
	<tr >
	    <td>blender_materials</td>
        <td>29.78</td> <td>30.73</td>
	</tr>
	<tr >
	    <td>blender_mic</td>
        <td>36.22</td> <td>34.05</td>
	</tr>
	<tr >
	    <td>blender_ship</td>
        <td>31.1</td> <td>30.0</td>
	</tr>
	<tr >
	    <td>average</td>
        <td>32.92</td> <td>32.58</td>
	</tr>
</table>



### HumanNeRF

#### Neural Body

<table>
	<tr>
	    <th rowspan="2">test data</th>
        <th colspan="2">PSNR</th>
        <th colspan="2">SSIM</th>
	</tr >
	<tr>
	    <th>Neural Body</th>
	    <th>XRNeRF</th>
	    <th>Neural Body</th>
	    <th>XRNeRF</th>
	</tr >
	<tr >
	    <td>313</td>
        <td>35.21</td> <td>37.76</td> <td>0.985</td> <td>0.993</td>
	</tr>
	<tr >
	    <td>315</td>
        <td>33.07</td> <td>35.99</td> <td>0.988</td> <td>0.992</td>
	</tr>
	<tr >
	    <td>377</td>
        <td>33.86</td> <td>33.86</td> <td>0.985</td> <td>0.986</td>
	</tr>
	<tr >
	    <td>386</td>
        <td>36.07</td> <td>34.24</td> <td>0.984</td> <td>0.984</td>
	</tr>
	<tr >
	    <td>387</td>
        <td>31.39</td> <td>31.99</td> <td>0.975</td> <td>0.979</td>
	</tr>
	<tr >
	    <td>390</td>
        <td>34.48</td> <td>35.45</td> <td>0.980</td> <td>0.984</td>
	</tr>
	<tr >
	    <td>392</td>
        <td>35.76</td> <td>35.11</td> <td>0.984</td> <td>0.986</td>
	</tr>
	<tr >
	    <td>393</td>
        <td>33.24</td> <td>33.50</td> <td>0.979</td> <td>0.985</td>
	</tr>
	<tr >
	    <td>394</td>
        <td>34.31</td> <td>35.61</td> <td>0.980</td> <td>0.984</td>
	</tr>
</table>


#### Animatable NeRF

<table>
	<tr>
	    <th rowspan="2">test data (Novel pose)</th>
        <th colspan="2">PSNR</th>
        <th colspan="2">SSIM</th>
	</tr >
	<tr>
	    <th>Animatable NeRF</th>
	    <th>XRNeRF</th>
	    <th>Animatable NeRF</th>
	    <th>XRNeRF</th>
	</tr >
	<tr >
	    <td>S1</td>
        <td>30.11</td> <td>31.98</td> <td>0.981</td> <td>0.984</td>
	</tr>
	<tr >
	    <td>S5</td>
        <td>32.60</td> <td>33.25</td> <td>0.987</td> <td>0.990</td>
	</tr>
	<tr >
	    <td>S6</td>
        <td>29.49</td> <td>30.12</td> <td>0.972</td> <td>0.974</td>
	</tr>
	<tr >
	    <td>S7</td>
        <td>31.54</td> <td>34.47</td> <td>0.984</td> <td>0.988</td>
	</tr>
	<tr >
	    <td>S8</td>
        <td>30.77</td> <td>32.01</td> <td>0.983</td> <td>0.985</td>
	</tr>
	<tr >
	    <td>S9</td>
        <td>31.94</td> <td>28.61</td> <td>0.980</td> <td>0.976</td>
	</tr>
	<tr >
	    <td>S11</td>
        <td>33.12</td> <td>33.43</td> <td>0.986</td> <td>0.986</td>
	</tr>
</table>


#### GNR

<table>
	<tr>
	    <th rowspan="2">test data</th>
        <th colspan="2">PSNR</th>
        <th colspan="2">SSIM</th>
	</tr >
	<tr>
	    <th>GNR</th>
	    <th>XRNeRF</th>
	    <th>GNR</th>
	    <th>XRNeRF</th>
	</tr >
	<tr >
	    <td>amanda</td>
        <td>23.62</td> <td>25.35</td> <td>0.93</td> <td>0.95</td>
	</tr>
	<tr >
	    <td>barry</td>
        <td>29.28</td> <td>30.71</td> <td>0.94</td> <td>0.95</td>
	</tr>
	<tr >
	    <td>fuzhizhi</td>
        <td>21.96</td> <td>21.42</td> <td>0.90</td> <td>0.89</td>
	</tr>
	<tr >
	    <td>jinyutong</td>
        <td>23.90</td> <td>24.08</td> <td>0.90</td> <td>0.91</td>
	</tr>
	<tr >
	    <td>joseph</td>
        <td>26.30</td> <td>24.46</td> <td>0.94</td> <td>0.92</td>
	</tr>
	<tr >
	    <td>maria</td>
        <td>21.51</td> <td>23.69</td> <td>0.90</td> <td>0.90</td>
	</tr>
	<tr >
	    <td>mahaoran</td>
        <td>28.41</td> <td>30.93</td> <td>0.93</td> <td>0.94</td>
	</tr>
	<tr >
	    <td>natacha</td>
        <td>28.71</td> <td>27.98</td> <td>0.91</td> <td>0.91</td>
	</tr>
	<tr >
	    <td>soufianou</td>
        <td>27.64</td> <td>28.83</td> <td>0.93</td> <td>0.93</td>
	</tr>
	<tr >
	    <td>zhuna</td>
        <td>25.40</td> <td>24.32</td> <td>0.93</td> <td>0.92</td>
	</tr>
</table>
