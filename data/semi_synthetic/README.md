# Clinical input data

`semi-syn-raw.csv` holds 20,000 hourly records drawn from the training data of the
PhysioNet/Computing in Cardiology Challenge 2019, *Early Prediction of Sepsis from Clinical Data*,
version 1.0.0 (https://physionet.org/content/challenge-2019/1.0.0/).

Changes made to the source data: a subset of rows was drawn, 17 of the source variables were kept
(Hour, HR, O2Sat, SBP, MAP, DBP, Resp, BaseExcess, pH, PaCO2, Glucose, Lactate, Potassium, Age,
HospAdmTime, ICULOS, SepsisLabel), and a column `SOFA_proxy_total` was added. The experiments use
six of these columns (Lactate, MAP, O2Sat, Resp, Age, pH); see `semi_synthetic.py` for how the
semi-synthetic outcome is generated from them.

## License

Contains information from *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing
in Cardiology Challenge 2019* (https://physionet.org/content/challenge-2019/1.0.0/), which is made
available here under the Open Database License (ODbL) v1.0
(https://opendatacommons.org/licenses/odbl/1-0/). `LICENSE-ODbL.txt` is the license file distributed
with the source data. The source project page lists the Creative Commons Attribution 4.0
International license (https://creativecommons.org/licenses/by/4.0/).

## Citation requested by the data provider

- Reyna, M., Josef, C., Jeter, R., Shashikumar, S., Moody, B., Westover, M. B., Sharma, A., Nemati, S.,
  and Clifford, G. D. (2019). Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in
  Cardiology Challenge 2019 (version 1.0.0). PhysioNet. https://doi.org/10.13026/v64v-d857
- Reyna, M. A., Josef, C. S., Jeter, R., Shashikumar, S. P., Westover, M. B., Nemati, S., Clifford, G. D.,
  and Sharma, A. Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology
  Challenge. Critical Care Medicine 48(2):210-217. https://doi.org/10.1097/CCM.0000000000004145
- Pollard, T., Moody, B. E., Lehman, L. H., Gow, B. J., Fernandes, C., Xie, C., Johnson, A., Mark, R. G.,
  and Heldt, T. (2026). PhysioNet as a global platform for biomedical research. Nature Health 1(8):792-795.
  https://doi.org/10.1038/s44360-026-00096-z
