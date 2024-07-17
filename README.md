# test that were made for multiprocessing

for 1000 frequencies (15 periods and 5 samples per periods, 1e-9 precision) :

| method    | time(s) |
|-----------|---------|
| map_async | 12.63   |
| map       | 12.85   |
| imap      | 12.85   |

so pretty much the same time for all the methods

## how to launch the code on windows

```bash
cd C:\Users\lab07\Documents\pybamm2
python -m venv ./venv/
.\venv\Scripts\activate.bat
pip install pybamm tqdm matplotlib
python EISdifferentSOC.py
```

and the next time :

```bash
cd C:\Users\lab07\Documents\pybamm2
.\venv\Scripts\activate.bat
python EISdifferentSOC.py
```