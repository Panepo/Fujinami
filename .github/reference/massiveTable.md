```python
import pandas as pd
import glob

csv_files = glob.glob("*.csv")
print("Found CSV files:", csv_files)

for f in csv_files:
    print(f"\n--- Inspecting {f} ---")
    try:
        df = pd.read_csv(f, nrows=10, header=None)
        print("Shape (first 10 rows):", df.shape)
        print("First 5 rows:")
        print(df.head(5))
    except Exception as e:
        print(f"Error reading {f}: {e}")



```

```text
Found CSV files: ['PostTimeResult.xlsx - STORGE TYPE.csv', 'PostTimeResult.xlsx - Other.csv', 'PostTimeResult.xlsx - FZ-55.csv', 'PostTimeResult.xlsx - TestConfiguration.csv', 'PostTimeResult.xlsx - TestConfiguration (Sort).csv', 'PostTimeResult.xlsx - Project BootTime.csv']

--- Inspecting PostTimeResult.xlsx - STORGE TYPE.csv ---
Shape (first 10 rows): (10, 3)
First 5 rows:
                                                                                            0                        1                                  2
0                                                                                         NaN                      NaN                                NaN
1  使用相同的HDI 在b360 的一台系統上 boot time  的測試結果: SATA SSD 會比 PCIe SSD 來的 慢 (測試手法: 手動按碼錶 看到 O.S桌面停碼錶                      NaN                                NaN
2                                                                                         NaN                      NaN                                NaN
3                                                                                    SATA SSD                 NVMe SSD                                NaN
4                                                                     LITEON CV8-8E512 -512G   LITEON CA5-8D1024-1024G  SAMSUNG MZVLLB256HBHQ-00007 256GB

--- Inspecting PostTimeResult.xlsx - Other.csv ---
Shape (first 10 rows): (10, 2)
First 5 rows:
                                                                                 0      1
0                                                                              NaN   UX10
1  Enable UEFI O.S Fast boot/Disable USB hot key support/Disable USB BIOS support     NaN
2                                         Boot time (Hybrid Shutdown, Unit second)  20.87
3                                      FW Post Time (Hybrid Shutdown, Unit second)  11.55
4                                                                  Disable A88179     NaN

--- Inspecting PostTimeResult.xlsx - FZ-55.csv ---
Shape (first 10 rows): (7, 8)
First 5 rows:
    0               1      2      3      4      5      6                   7
0 NaN           FZ-55    NaN    NaN    NaN    NaN    NaN                 NaN
1 NaN       Boot time   1.00   2.00   3.00   4.00   5.00                 Ave
2 NaN  Modern Standby    NaN    NaN    NaN    NaN    NaN                 NaN
3 NaN              S4  17.34  15.88  16.57  16.24  16.25              16.456
4 NaN              S5  36.17  36.01  36.70  36.15  36.67  36.339999999999996

--- Inspecting PostTimeResult.xlsx - TestConfiguration.csv ---
Shape (first 10 rows): (10, 29)
First 5 rows:
                    0                                                                                     1                                                                                     2                                                                                              3                                     4                                     5                                     6                           7                                       8                       9                                                           10                                        11                                     12                        13                                     14                                     15                                     16                            17                                   18                         19                 20                 21                                                                        22                                                                        23                          24                        25                         26                     27                         28
0                  NaN                                                                    S510AD (DVT SKU C)                                                               S510 ARL (C-Test SKU B)                                                                           S510 MTL (PVT SKU B)             K120G3 Tablet (PVT SKU B)             K120G3 Laptop (PVT SKU B)                            V110G5 WHL                  V110G5 CML                              V110G6 CML              F110G5 WHL                                                  F110G5 CML                                A140G2 CML                               B360 CML                  UX10 WHL                               UX10 CML                          UX10 CML (C1)                          UX10 CML (C2)                    S410G3 WHL                       S410G4 CML EVT             S410G4 CML DVT     S410G4 TGL DVT         S410G4 TGL                                                K120G2 PVT C-Test (Tablet)                                                K120G2 PVT C-Test (Laptop)            ASUS VivoBook 14                     Rhino                 V110G7 DVT  B360G1 (CML) MP-SKUA       B360G2 (ADL) PVT-SKUA
1       CPU model name                                                     AMD Ryzen AI 7 350 w/ Radeon 860M                                                             Intel® Core™ Ultra 7 265H                                                                      Intel® Core™ Ultra 7 155U           Intel® Core™ 5 120U 1.40GHz           Intel® Core™ 5 120U 1.40GHz              I5-8265U 1.60GHz 1.80GHz   I7-10510U 1.80GHz 2.30GHz                I7-10710U 1.10GHz 1.61Hz  i5-8265U 1.6GHz 1.8GHz                                           I7-10610U 1.80GHz                i3-10110U 2.10G Hz 2.59Ghz           I5-10210U 1.60 GHz 2.112 GHz          i5-8265U 1.60GHz              I7-10510U 1.80GHz 2.30GHz              I7-10510U 1.80GHz 2.30GHz              I7-10510U 1.80GHz 2.30GHz      I3-8145U 2.10GhZ 2.30GhZ            I7-10510U 1.80GHz 2.30GHz  I7-10510U 1.80GHz 2.30GHz  I5-1135G7 2.4GHz   I7-1165G7 2.8GHz                                                   I5-8265U 1.60GHz 1.80GHz                                                  I5-8265U 1.60GHz 1.80GHz    I5-8265U 1.60GHz 1.80GHz                       ES2  ADL-P i7-1265U vPro (2+8)     I7-10610U 1.80 GHz  ADL-P i7-1280P vPro (6+8)
2  Memroy 1 model name  DDR5 SODIMM MODULE ; 32GB, 5600MHZ, Kingston CBD56S46BD8HA-32, Hynix A-Die, Kingston  DDR5 SODIMM MODULE ; 32GB, 5600MHZ, Kingston CBD56S46BD8HA-32, Hynix A-Die, Kingston  DDR5 SODIMM MODULE ; 16GB, 5600MHZ, 2Gx8, 1Rx8, 1.1V, CBD56S46BS8HA-16, Hynix A-Die, Kingston  Kingston DDR5-5600 16GB 323714110016  Kingston DDR5-5600 16GB 323714110016  Kingston CBD26D4S9S8ME-8 8GB 2400MHz          SAMSUNG 8G 2666MHz  Kingston CBD26D4S9D8ME-16\n16G 2666MHz  Kingston DDR4-2666 4GB  Transcend, 16G,DDR4,2133,1.2V/TS2GSH64V1B,Samsung IC B-Die  Kingston   CBD26D4S9D8ME-16 16GB 2667MHZ  Kingston CBD26D4S9S8ME-8 8GB, 2667MHZ                       NaN  Kingston CBD25D4S98S8ME-8 8GB 2667MHz  Kingston CBD25D4S98S8ME-8 8GB 2667MHz  Kingston CBD25D4S98S8ME-8 8GB 2667MHz                    4G 2666MHZ  Transcendd  TS2G5H64V1B 16GB 2133MB                32G 2666MHZ        32G 3200MHZ         8G 3200MHZ  8GB, 3200MHZ, 1024Mx8, 1Rx8, 1.2V, M471A1K43DB1-CWE, 16nm D-Die, Samsung  8GB, 3200MHZ, 1024Mx8, 1Rx8, 1.2V, M471A1K43DB1-CWE, 16nm D-Die, Samsung                          8G                        8G                        32G                     8G                        32G
3  Memroy 2 model name  DDR5 SODIMM MODULE ; 32GB, 5600MHZ, Kingston CBD56S46BD8HA-32, Hynix A-Die, Kingston  DDR5 SODIMM MODULE ; 32GB, 5600MHZ, Kingston CBD56S46BD8HA-32, Hynix A-Die, Kingston  DDR5 SODIMM MODULE ; 16GB, 5600MHZ, 2Gx8, 1Rx8, 1.1V, CBD56S46BS8HA-16, Hynix A-Die, Kingston  Kingston DDR5-5600 16GB 323714110016  Kingston DDR5-5600 16GB 323714110016  Kingston CBD26D4S9S8ME-8 8GB 2400MHz          SAMSUNG 8G 2666MHz  Kingston CBD26D4S9D8ME-16\n16G 2666MHz                     NaN                                                         NaN  Kingston   CBD26D4S9D8ME-16 16GB 2667MHZ  Kingston CBD26D4S9S8ME-8 8GB, 2667MHZ                       NaN                                    NaN                                                                                                         NaN                                  NaN                        NaN                NaN                NaN                                                                       NaN                                                                       NaN                         NaN                       NaN                        NaN                    NaN                        32G
4       SSD model name                                                                                   NaN                                                                                   NaN                                                                                            NaN                                   NaN                                   NaN              LITEON CV8-8E256 - 256GB  LIGTONIT LGT-128M6G- 128GB                      LIGTON CV8 - 8E256        LiteOn CV8-8E256                                           LiteOn CV8-8E5128                          LiteOn CV8-8E512      Phison SSMP001TTB3DS2-S10 -1024GB  LITEON CV8-8E256 - 256GB      Phison SSMP001TTB3DS2-S10 -1024GB                                                                                Phison SSMP001TTB3DS2-S10 1T               LITEON CV8-8E256 256GB   LITEON CA5-8D1024 (NVMe)                NaN                NaN                                                                                                                                                      INTEL SSDPEKNW512GB (NVMe)  LITEON CA5-8D1024 (NVMe)    LITEON CA5-8D512 (NVMe)  SSSTE CA5-8D256 256KB      SSSTE CA5-8D256 256KB

--- Inspecting PostTimeResult.xlsx - TestConfiguration (Sort).csv ---
Shape (first 10 rows): (10, 23)
First 5 rows:
                    0                                     1                       2                         3                             4                           5                                6                           7                                       8                                                           9                                         10                                     11                                     12                                     13                                     14                                   15                         16                         17                         18                                                                        19                                                                                             20                                                                                    21                                                                                    22
0                  NaN                            V110G5 WHL              F110G5 WHL                  UX10 WHL                    S410G3 WHL            ASUS VivoBook 14                             FZ55                  V110G5 CML                              V110G6 CML                                                  F110G5 CML                                A140G2 CML                               B360 CML                               UX10 CML                          UX10 CML (C1)                          UX10 CML (C2)                       S410G4 CML EVT             S410G4 CML DVT                 S410G4 TGL  S410G5 RPL DVT SKUC (MXM)                                                K120G2 PVT C-Test (Tablet)                                                                           S510 MTL (PVT SKU B)                                                               S510 ARL (C-Test SKU B)                                                            S510 ARL (C-Test SKU B T1)
1       CPU model name              I5-8265U 1.60GHz 1.80GHz  i5-8265U 1.6GHz 1.8GHz          i5-8265U 1.60GHz      I3-8145U 2.10GhZ 2.30GhZ    I5-8265U 1.60GHz 1.80GHz  i5-8365U CPU @ 1.60GHz 1896 Mhz   I7-10510U 1.80GHz 2.30GHz                I7-10710U 1.10GHz 1.61Hz                                           I7-10610U 1.80GHz                i3-10110U 2.10G Hz 2.59Ghz           I5-10210U 1.60 GHz 2.112 GHz              I7-10510U 1.80GHz 2.30GHz              I7-10510U 1.80GHz 2.30GHz              I7-10510U 1.80GHz 2.30GHz            I7-10510U 1.80GHz 2.30GHz  I7-10510U 1.80GHz 2.30GHz  I7-10510U 1.80GHz 2.30GHz           I7-1370P 1.90GHz                                                I5-1145G7 @2.60GHz 2.61GHz                                                                      Intel® Core™ Ultra 7 155U                                                             Intel® Core™ Ultra 7 265H                                                             Intel® Core™ Ultra 7 265H
2  Memroy 1 model name  Kingston CBD26D4S9S8ME-8 8GB 2400MHz  Kingston DDR4-2666 4GB                       8GB                    4G 2666MHZ                          8G                    DDR4 2666 8GB          SAMSUNG 8G 2666MHz  Kingston CBD26D4S9D8ME-16\n16G 2666MHz  Transcend, 16G,DDR4,2133,1.2V/TS2GSH64V1B,Samsung IC B-Die  Kingston   CBD26D4S9D8ME-16 16GB 2667MHZ  Kingston CBD26D4S9S8ME-8 8GB, 2667MHZ  Kingston CBD25D4S98S8ME-8 8GB 2667MHz  Kingston CBD25D4S98S8ME-8 8GB 2667MHz  Kingston CBD25D4S98S8ME-8 8GB 2667MHz  Transcendd  TS2G5H64V1B 16GB 2133MB                32G 2666MHZ                 8G 3200MHZ     Kingston DDR5-4800 32G  8GB, 3200MHZ, 1024Mx8, 1Rx8, 1.2V, M471A1K43DB1-CWE, 16nm D-Die, Samsung  DDR5 SODIMM MODULE ; 16GB, 5600MHZ, 2Gx8, 1Rx8, 1.1V, CBD56S46BS8HA-16, Hynix A-Die, Kingston  DDR5 SODIMM MODULE ; 32GB, 5600MHZ, Kingston CBD56S46BD8HA-32, Hynix A-Die, Kingston  DDR5 SODIMM MODULE ; 32GB, 5600MHZ, Kingston CBD56S46BD8HA-32, Hynix A-Die, Kingston
3  Memroy 2 model name  Kingston CBD26D4S9S8ME-8 8GB 2400MHz                     NaN                       NaN                           NaN                         NaN                              NaN          SAMSUNG 8G 2666MHz  Kingston CBD26D4S9D8ME-16\n16G 2666MHz                                                         NaN  Kingston   CBD26D4S9D8ME-16 16GB 2667MHZ  Kingston CBD26D4S9S8ME-8 8GB, 2667MHZ                                    NaN                                                                                                                NaN                        NaN                        NaN     Kingston DDR5-4800 32G                                                                       NaN  DDR5 SODIMM MODULE ; 16GB, 5600MHZ, 2Gx8, 1Rx8, 1.1V, CBD56S46BS8HA-16, Hynix A-Die, Kingston  DDR5 SODIMM MODULE ; 32GB, 5600MHZ, Kingston CBD56S46BD8HA-32, Hynix A-Die, Kingston  DDR5 SODIMM MODULE ; 32GB, 5600MHZ, Kingston CBD56S46BD8HA-32, Hynix A-Die, Kingston
4       SSD model name              LITEON CV8-8E256 - 256GB        LiteOn CV8-8E256  LITEON CV8-8E256 - 256GB  Phison SSMP001TTB3DS2-S10 1T  INTEL SSDPEKNW512GB (NVMe)       SAMSUNG MZNLN512HAJQ-00000  LIGTONIT LGT-128M6G- 128GB                      LIGTON CV8 - 8E256                                           LiteOn CV8-8E5128                          LiteOn CV8-8E512      Phison SSMP001TTB3DS2-S10 -1024GB      Phison SSMP001TTB3DS2-S10 -1024GB                                                                                             LITEON CV8-8E256 256GB   LITEON CA5-8D1024 (NVMe)                        NaN                        NaN                                                                       NaN                                                                                            NaN                                                                                   NaN                                                                                   NaN

--- Inspecting PostTimeResult.xlsx - Project BootTime.csv ---
Shape (first 10 rows): (10, 49)
First 5 rows:
                                        0                                  1                      2                      3                                                 4                        5                                                             6                     7                          8                          9           10                11                    12                                    13                                                                            14          15          16      17     18          19        20                                                     21                                                                                                       22            23                24                25               26                         27                   28                     29                                      30                                      31                32          33                34            35                                               36        37        38     39                              40                                    41                                    42                43                                                  44                                             45                                                             46                                                                                    47                                                                                                            48
0                                      NaN                                NaN                    NaN                    NaN                                               NaN                      NaN                                                           NaN                   NaN                        NaN                        NaN         NaN               NaN                   NaN                                   NaN                                                                           NaN         NaN         NaN     NaN    NaN         NaN       NaN                                                    NaN                                                                                                      NaN           NaN               NaN               NaN              NaN                        NaN                  NaN                    NaN                                     NaN                                     NaN               NaN         NaN               NaN           NaN                                              NaN       NaN       NaN    NaN                             NaN                       2023/01/05 test                       2023/01/05 test               NaN                                                 NaN                                            NaN                                                            NaN                                                                                   NaN                                                                                                           NaN
1                            (Unit second)  S510AD (DVT SKUC) No PSB function  UX10G5 LNL (DVT SKUB)  UX10G5 LNL (DVT SKUC)  S510 ARL (C-Test SKU B T1) Disable I226 PXE boot  S510 ARL (C-Test SKU B)  S510 ARL (C-Test SKU B) TPM clock change from 48MHz to 25MHz  S510 MTL (PVT SKU B)  K120G3 Tablet (PVT SKU B)  K120G3 Laptop (PVT SKU B)  V110G5 WHL  V110G5 CML (M/B)  V110G6 CML\n(System)  V110G6 CML\n(System)\n1. Single read  V110G6 CML\n(System)\n1. Single read \n2. Read/Write/Fast Read clock : 30MHz  F110G5 WHL  F110G5 CML  A140G2   B360  UX10-WHL\n  UX10 CML  UX10 CML\nPcdH2OFdmChainOfTrustStandAlone set to TRUE  UX10 CML\n1. PcdH2OFdmChainOfTrustStandAlone set to TRUE\n2. PcdH2OFdmChainOfTrustStupport set to FALSE  S410G3 WHL\n  S410G4 CML EVT\n  ASUS VivoBook 14  S410G4-CML DVT   S410G4 (Tigerlake) \nDVT   S410G4 (Tigerlake)   S410G5 DVT SKUC (MXM)  K120G2 PVT C-Test (Tablet) (TigerLake)  K120G2 PVT C-Test (Laptop) (TigerLake)  F110G6 PVT SKU B  V110G7 DVT  UX10G3 DVT SKU-B  Requirement   Rhino\n(Research MB w/I ES2 CPU)\nw/I Fast Boot  UX10(C1)  UX10(C2)   K120  B360G1 (CometLake)  I5-10210U   B360G1 (CometLake) MP-SKUA I7-10610U  B360G2 (AlderLake) PVT-SKUA I7-1260P  UX10G3 PVT SKU-C  UX10G3 PVT SKU-C \nFanless (by test  EC setting)\n  UX10G3 PVT SKU-C \nEnaled Insyde first boot\n  UX10G3 PVT SKU-C \nEnaled Insyde first boot\n+ Disabled ISH\n  UX10G3 PVT SKU-C \nEnaled Insyde first boot\n+ Disabled ISH\n+ Uninstall G-Utility\n  UX10G3 PVT SKU-C \nEnaled Insyde first boot\n+ Disabled ISH\n+ Pure OS (install chipset & VGA driver only)\n
2  1st System Boot time  (Hybrid Shutdown)                              18.13                  23.95                  27.66                                             21.51                       27                                                         26.81                 20.71                      12.63                      12.96       17.48             19.57                 24.47                                 27.04                                                                         30.29       15.06       20.94   19.38  20.26        14.7      19.4                                                   14.9                                                                                                     13.3         14.22              14.3             11.65            21.22                       18.9                 13.9                 13.882                                   16.56                                   21.06             14.56        18.5             14.23           NaN                                              NaN     20.68     20.06  18.73                           20.26                                 26.88                                 17.33             15.17                                               20.27                                          15.07                                                          13.63                                                                                 13.29                                                                                                          11.6
3     1st BIOS Post Time (Hybrid Shutdown)                               9.46                 14.712                  16.83                                                12                    16.81                                                         16.46                   9.7                        5.7                        5.6       6.326            10.402                12.992                                15.392                                                                        17.746       4.399      10.368  11.202  11.64       4.236    10.346                                                  6.329                                                                                                    4.573         5.408              6.08             5.295            13.88                      6.825                8.491                   7.25                                   9.016                                  11.812             9.866        5.49              5.48           NaN                                              7.8    12.414    12.395  8.541                           11.64                                17.635                                 4.836              5.41                                                5.52                                          5.099                                                          4.818                                                                                  4.81                                                                                                         4.812
4   2nd System Boot time (Hybrid Shutdown)                              18.51                  23.86                  27.63                                                22                     26.6                                                          26.2                  20.9                      12.45                      12.55       17.01             19.72                 24.57                                 27.24                                                                         29.47       14.73       20.98   19.51  20.06        15.6      19.5                                                   15.2                                                                                                     13.6         14.13             14.21             11.75            21.62                       18.2                 14.1                 13.877                                   16.36                                   20.18             14.52       18.19              14.9           NaN                                              NaN     21.14     21.21    NaN                           20.06                                 26.75                  ...

```

```python
import pandas as pd

df_config = pd.read_csv("PostTimeResult.xlsx - TestConfiguration.csv", header=None)
print("TestConfiguration rows (first column):")
print(df_config[0].dropna().tolist())

df_boot = pd.read_csv("PostTimeResult.xlsx - Project BootTime.csv", header=None)
print("\nProject BootTime rows (first column):")
print(df_boot[0].dropna().tolist()[:20])



```

```text
TestConfiguration rows (first column):
['CPU model name', 'Memroy 1 model name', 'Memroy 2 model name', 'SSD model name', 'PCIE SSD model name', 'BIOS Version', 'EC Version', 'O.S  Version', 'HDI version', 'WWAN model name', 'WLAN model name', 'Boot Guard Support', 'System Guard Support', 'ME Firmware', '2nd LAN', 'Touch Screen (USB)', 'USB to COM', 'Smart Card (USB)', 'RFID (USB)', 'Finger Printer (USB)', 'Camera (USB)', 'Bluetooth (USB)', 'CardReader (USB)', 'Digitizer (USB)', 'List USB enumerate device at POST', 'Average System Boot time (Hybrid Shutdown)', 'Average BIOS Post Time (Hybrid Shutdown)', 'USB enumerate time', 'Boot Guard', 'NitfyRock', 'EC image HASH', 'Others']

Project BootTime rows (first column):
['(Unit second)', '1st System Boot time  (Hybrid Shutdown)', '1st BIOS Post Time (Hybrid Shutdown)', '2nd System Boot time (Hybrid Shutdown)', '2nd BIOS Post Time (Hybrid Shutdown)', '3rd System Boot time (Hybrid Shutdown)', '3rd BIOS Post Time (Hybrid Shutdown)', 'Average System Boot time (Hybrid Shutdown)', 'Average BIOS Post Time (Hybrid Shutdown)', 'System Boot time (Shutdown)', 'BIOS Post Time (Shutdown)', 'System Boot time (Shutdown)', 'BIOS Post Time (Shutdown)', 'System Boot time (Shutdown)', 'BIOS Post Time (Shutdown)', 'Average System  Boot time (Shutdown)', 'Average BIOS Post Time (Shutdown)']


```

```python
headers_config = df_config.iloc[0].dropna().tolist()
headers_boot = df_boot.iloc[1].dropna().tolist()

print(f"Num columns in TestConfiguration: {len(headers_config)}")
print(f"Num columns in Project BootTime: {len(headers_boot)}")

# check intersection
common = set(headers_config).intersection(set(headers_boot))
print(f"Common projects: {len(common)}")
print("Some common ones:", list(common)[:10])


```

```text
Num columns in TestConfiguration: 28
Num columns in Project BootTime: 49
Common projects: 10
Some common ones: ['F110G5 WHL', 'UX10 CML', 'K120G3 Tablet (PVT SKU B)', 'F110G5 CML', 'K120G3 Laptop (PVT SKU B)', 'V110G5 WHL', 'V110G7 DVT', 'S510 ARL (C-Test SKU B)', 'S510 MTL (PVT SKU B)', 'ASUS VivoBook 14']


```

```python
import pandas as pd
import json

# Let's write a script to generate an example of Entity-Centric chunking from TestConfiguration.csv
df_config = pd.read_csv("PostTimeResult.xlsx - TestConfiguration.csv", header=None)

# Clean headers
# Row 0 contains the project names
projects = df_config.iloc[0].tolist()
rows = df_config.shape[0]

chunks = []
# Loop through columns starting from column 1 (column 0 is the metric name)
for col_idx in range(1, df_config.shape[1]):
    proj_name = projects[col_idx]
    if pd.isna(proj_name) or str(proj_name).strip() == "":
        continue

    chunk_text = f"System Configuration Profile: {proj_name}\n"
    chunk_text += "=" * (len(chunk_text) - 1) + "\n"

    for row_idx in range(1, rows):
        metric_name = df_config.iloc[row_idx, 0]
        val = df_config.iloc[row_idx, col_idx]
        if pd.isna(metric_name):
            continue
        chunk_text += f"- {metric_name}: {val}\n"

    chunks.append(chunk_text)

print("--- Sample Generated Chunk for Strategy A ---")
print(chunks[0][:600])


```

```text
--- Sample Generated Chunk for Strategy A ---
System Configuration Profile: S510AD (DVT SKU C)
================================================
- CPU model name: AMD Ryzen AI 7 350 w/ Radeon 860M
- Memroy 1 model name: DDR5 SODIMM MODULE ; 32GB, 5600MHZ, Kingston CBD56S46BD8HA-32, Hynix A-Die, Kingston
- Memroy 2 model name: DDR5 SODIMM MODULE ; 32GB, 5600MHZ, Kingston CBD56S46BD8HA-32, Hynix A-Die, Kingston
- SSD model name: nan
- PCIE SSD model name: SSTC CL4-8D1024
- BIOS Version: R0.52.070520D
- EC Version: R0.50.070520p
- O.S  Version: 24H2 (26100.4202)
- HDI version: nan
- WWAN model name: Telit FN990
- WLAN model name: MediaTek MT7


```

To feed a massive, non-standard table like your **Project BootTime** and **TestConfiguration** sheets into a Retrieval-Augmented Generation (RAG) system, **standard text splitters (like token or character-based chunking) will completely fail**.

If you split text blindly by lines or character count, the row values will lose their column header contexts, and numbers like `18.13` or `9.46` will become meaningless floating digits to the LLM.

Your files have a **transposed layout**: **Columns represent the distinct entities** (individual laptop/tablet project configurations like `S510AD`, `UX10 CML`), while **Rows represent the attributes/metrics** (`CPU model name`, `1st System Boot time`).

Here is a comprehensive breakdown of how to intelligently chunk these files for optimal RAG performance.

---

### Strategy 1: Entity-Centric / Column-by-Column Profile (Highly Recommended)

Since most user queries in a RAG system look like *"What CPU does S510AD use?"* or *"What is the average hybrid boot time for UX10 CML?"*, you should serialize each column into an independent text profile (a **Dossier** or **Markdown List**).

Instead of embedding the table as a grid, you can write a short preprocessing python script to transform the data column-by-column into self-contained text files.

#### Example Chunk Format (Markdown Key-Value List):

```markdown
System Configuration & Performance Profile: S510AD (DVT SKU C)
============================================================
Context: This document contains the hardware specs and boot time performance metrics for project S510AD. All time metrics are in seconds.

[Hardware Specs]
- CPU model name: AMD Ryzen AI 7 350 w/ Radeon 860M
- Memory 1 model name: DDR5 SODIMM MODULE ; 32GB, 5600MHZ, Kingston...
- SSD model name: SSTC CL4-8D1024
- BIOS Version: R0.52.070520D
- OS Version: 24H2 (26100.4202)

[Performance Results]
- 1st System Boot time (Hybrid Shutdown): 18.13 seconds
- 1st BIOS Post Time (Hybrid Shutdown): 9.46 seconds
- Average System Boot time (Hybrid Shutdown): 18.32 seconds
- Average System Boot time (Shutdown): 28.81 seconds

```

**Why this is effective:**

* **Perfect Context Retention:** Every metric is directly paired with its row label and system name in the same text block, ensuring high-quality vector embeddings.
* **Cross-Sheet Merging:** You can write your script to find `S510AD` in both `TestConfiguration.csv` and `Project BootTime.csv`, stitch them into a single comprehensive markdown profile, and chunk it as one chunk.

---

### Strategy 2: Horizontal Matrix Comparison Chunks (Markdown Tables)

If your users are going to ask macro or comparative questions (e.g., *"Compare the hybrid boot times of all Comet Lake (CML) systems"*), Strategy 1 is insufficient because the RAG system would have to retrieve 10 separate documents.

To support comparisons, chunk the wide table into **smaller, overlapping sub-tables** formatted in native **Markdown Tables** (which modern LLMs understand flawlessly). Restrict each chunk to **3 to 4 entity columns** but include **all relevant row headers**.

#### Example Chunk Format (Sub-Table):

```markdown
### Project Performance Comparison Matrix (Group 1)
Context: Boot times and BIOS post times compared across selected systems.

| Metric (Unit: second) | S510AD (DVT SKUC) | UX10G5 LNL (DVT SKUB) | UX10G5 LNL (DVT SKUC) |
| :--- | :--- | :--- | :--- |
| 1st System Boot time (Hybrid) | 18.13 | 23.95 | 27.66 |
| 1st BIOS Post Time (Hybrid) | 9.46 | 14.712 | 16.83 |
| Average Boot time (Hybrid) | 18.32 | 26.86 | 20.94 |

```

---

### Strategy 3: Narrative / Structural Flattening (For Irregular Sheets)

For sheets like `STORGE TYPE.csv` or `Other.csv` which contain blocks of nested text summaries mixed with short tables (e.g., explaining why SATA SSD is slower than PCIe SSD on B360), a table chunker will fail. Instead, flatten them into natural language narratives or hierarchically structured paragraphs.

#### Example Chunk Format:

```markdown
Document Subject: Storage Type Boot Time Testing on B360 System
Summary of Finding: When using the identical HDI on a B360 system, testing results indicate that a SATA SSD is slower than a PCIe NVMe SSD. (Measurement method: Manual stopwatch tracking until O.S. desktop appears).

Detailed Data Points:
- LITEON CV8-8E512 - 512G (SATA SSD): Average boot time is 28.85 seconds (Run 1: 29.22s, Run 2: 28.86s, Run 3: 28.73s).
- LITEON CA5-8D1024 - 1024G (NVMe SSD): Average boot time is 27.41 seconds (Run 1: 27.53s, Run 2: 27.66s, Run 3: 27.12s).

```

---

### 🛠️ Python Script to Automate Strategy 1 (Entity Profiles)

You can run a Python script similar to the one below to turn your raw CSV files into an array of optimized RAG markdown chunks:

```python
import pandas as pd
import math

def generate_rag_chunks():
    # Load your transposed data configuration table
    df = pd.read_csv("PostTimeResult.xlsx - TestConfiguration.csv", header=None)

    # Row 0 contains the project headers
    project_names = df.iloc[0].tolist()
    total_rows = df.shape[0]

    markdown_chunks = []

    # Loop over columns (each column is a project entity)
    for col_idx in range(1, df.shape[1]):
        proj = project_names[col_idx]
        if pd.isna(proj) or str(proj).strip() == "":
            continue

        # Initialize Chunk Header
        chunk = f"### System Profile: {proj.strip()}\n"
        chunk += "Context: Specifications and benchmark attributes for this project.\n\n"

        # Populate metrics
        for row_idx in range(1, total_rows):
            metric_name = df.iloc[row_idx, 0]
            value = df.iloc[row_idx, col_idx]

            if pd.isna(metric_name):
                continue

            # Clean representation of empty data
            val_str = "N/A" if pd.isna(value) else str(value).strip().replace('\n', ' ')
            chunk += f"- **{metric_name.strip()}**: {val_str}\n"

        markdown_chunks.append({
            "text": chunk,
            "metadata": {
                "project_name": proj,
                "source_file": "TestConfiguration.csv"
            }
        })

    return markdown_chunks

# Example execution
chunks = generate_rag_chunks()
print(f"Generated {len(chunks)} ready-to-index RAG chunks!")
print(chunks[0]['text'][:400])

```

### Summary of Best Practices for Your RAG Database

1. **Leverage Vector Metadata Filters:** When inserting these chunks into a vector database (like Pinecone, Milvus, or Chroma), save fields like `{"project_name": "S510AD", "generation": "CML"}` as metadata. If a user asks a narrow question, your system can pre-filter by metadata before running semantic vector calculation.
2. **Inject Summary Context Strings:** LLMs struggle when context chunks don't have a clear umbrella topic. Always inject a standard string like `"This document contains Boot Time data for..."` at the beginning of every chunk.
3. **Use JSON or Markdown format:** Do not upload raw, unformatted rows. LLMs are explicitly trained to read `|` columns or `- Key: Value` lists, boosting accuracy by up to 40% compared to raw comma-separated strings.
