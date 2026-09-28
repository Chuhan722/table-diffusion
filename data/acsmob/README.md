# acsmob（ACSmobility-CA 2018）

folktables ACS 2018 1-Year 加州 ACSMobility 任务，GSD 论文（ICML 2023）主场基准之一。20 特征加 MIG 标签共 21 列整数码，口径见 scripts/prep_acsmob.py 档头。行数 80329 与论文 64263 之差系 folktables 版本缺失值处理演化，两家基线同表公平。

原始列名对照：attr_1=AGEP，attr_2=SCHL，attr_3=MAR，attr_4=SEX，attr_5=DIS，attr_6=CIT，attr_7=MIL，attr_8=ANC，attr_9=NATIVITY，attr_10=RELP，attr_11=DEAR，attr_12=DEYE，attr_13=DREM，attr_14=RAC1P，attr_15=GCL，attr_16=COW，attr_17=ESR，attr_18=WKHP，attr_19=JWMNP，attr_20=PINCP，attr_21=MIG
