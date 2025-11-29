# Train/Test Split Summary

- generated_at: 2025-10-11 11:59:08

- seed: 42

## diabetes
- source: `data/clean/diabetes_clean.csv`
- target: **Outcome**
- rows(clean): 768
- train/test: 614 / 154 (test_size=0.2)
- class(train): `{0: 400, 1: 214}`
- class(test):  `{0: 100, 1: 54}`
- checksum(train): `377284881d5d2dfa5a0de98a507efe63`
- checksum(test):  `5790ee34f2aea6b59a419ffa6867970c`

## heloc
- source: `data/clean/heloc_clean.csv`
- target: **RiskPerformance**
- rows(clean): 9872
- train/test: 7897 / 1975 (test_size=0.2)
- class(train): `{'RIS_0': 4108, 'RIS_1': 3789}`
- class(test):  `{'RIS_0': 1028, 'RIS_1': 947}`
- checksum(train): `f8b38260211407fd8c1ecda1b129d895`
- checksum(test):  `54503def4fb4a402b86bba1dbf609d53`

## income
- source: `data/clean/income_clean.csv`
- target: **income**
- rows(clean): 48790
- train/test: 39032 / 9758 (test_size=0.2)
- class(train): `{'INC_0': 29687, 'INC_1': 9345}`
- class(test):  `{'INC_0': 7422, 'INC_1': 2336}`
- checksum(train): `110e9f4f0e06f1e0fa00487c789cf68d`
- checksum(test):  `bb1d9a8c6b6367f6104e189c957f091e`

## sick
- source: `data/clean/sick_clean.csv`
- target: **binaryClass**
- rows(clean): 3711
- train/test: 2968 / 743 (test_size=0.2)
- class(train): `{'BIN_0': 2735, 'BIN_1': 233}`
- class(test):  `{'BIN_0': 685, 'BIN_1': 58}`
- checksum(train): `f57c774de90964d87555deb938ea9d0b`
- checksum(test):  `0d6c6fd3d73480f9dbdc6f454543a8ed`

## thyroid
- source: `data/clean/thyroid_clean.csv`
- target: **Risk**
- rows(clean): 364
- train/test: 291 / 73 (test_size=0.2)
- class(train): `{'RIS_0': 184, 'RIS_1': 81, 'RIS_2': 26}`
- class(test):  `{'RIS_0': 46, 'RIS_1': 21, 'RIS_2': 6}`
- checksum(train): `d1212b2f0d18a9b9a34ab17216833f87`
- checksum(test):  `96d44f1fd88db84c4970fc5d100d58b5`

## travel
- source: `data/clean/travel_clean.csv`
- target: **TravelInsurance**
- rows(clean): 1987
- train/test: 1589 / 398 (test_size=0.2)
- class(train): `{0: 1021, 1: 568}`
- class(test):  `{0: 256, 1: 142}`
- checksum(train): `42edce9580be2644beae34c813bef360`
- checksum(test):  `ae103ca655b676ab7339bf1d1d2d320c`
