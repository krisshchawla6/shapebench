# Rewards where BO has the best median

**15 rewards** (7-method for NeuralFoil/BlendedNet, 6-method for Superwing)

| env | reward | 2nd best | BO margin | BO | lbfgsb | GA | cmaes | v3 | shinka | openevolve |
|---|---|---|---|---|---|---|---|---|---|---|
| NeuralFoil | low_re_multipoint | lbfgsb | 1.3438 | -3.9175 | -5.2612 | -10.0 | -10.0 | -10.0 | -10.0 | -10.0 |
| NeuralFoil | reward_exact_notebook | lbfgsb | 1.1737 | -3.9368 | -5.1105 | -10.0 | -10.0 | -10.0 | -10.0 | -10.0 |
| NeuralFoil | reward_hpa_endurance_weighted | lbfgsb | 1.1737 | -3.9368 | -5.1105 | -10.0 | -10.0 | -10.0 | -10.0 | -10.0 |
| NeuralFoil | reward_hpa_high_cl | lbfgsb | 1.1737 | -3.9368 | -5.1105 | -9.5682 | -10.0 | -10.0 | -10.0 | -10.0 |
| NeuralFoil | reward_hpa_low_cl | lbfgsb | 1.1737 | -3.9368 | -5.1105 | -10.0 | -10.0 | -10.0 | -10.0 | -10.0 |
| NeuralFoil | reward_hpa_mid_cl_unequal | lbfgsb | 1.1737 | -3.9368 | -5.1105 | -8.5594 | -10.0 | -10.0 | -10.0 | -10.0 |
| NeuralFoil | reward_hpa_strict | lbfgsb | 1.1737 | -3.9368 | -5.1105 | -10.0 | -10.0 | -10.0 | -10.0 | -10.0 |
| Superwing | sw_001 | GA | 230.1008 | 290.3583 | 50.044 | 60.2575 | — | 13.5149 | 8.4716 | 9.9708 |
| Superwing | sw_002 | GA | 234.2877 | 290.3583 | 50.044 | 56.0706 | — | 12.0632 | 6.7122 | 8.4513 |
| Superwing | sw_011 | lbfgsb | 49.6879 | 87.7756 | 38.0877 | 27.2956 | — | 24.6374 | 23.7629 | 33.338 |
| Superwing | sw_012 | GA | 2.3656 | 15.4214 | -5.0 | 13.0558 | — | -5.0 | 7.7268 | 12.2301 |
| Superwing | sw_014 | lbfgsb | 2.6077 | 11.4093 | 8.8016 | 8.3238 | — | 7.0444 | 6.1319 | 6.4055 |
| Superwing | sw_023 | openevolve | 3.1788 | 15.2385 | 11.8803 | 11.8555 | — | 9.5425 | 10.0929 | 12.0597 |
| Superwing | sw_024 | lbfgsb | 2.4 | 20.4753 | 18.0754 | 10.9284 | — | 11.1434 | 12.941 | 13.3308 |
| Superwing | sw_025 | lbfgsb | 4.5163 | 21.5858 | 17.0695 | 10.1259 | — | 10.9257 | 12.537 | 12.1434 |