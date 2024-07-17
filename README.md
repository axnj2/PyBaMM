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
C:\Users\lab07> cd C:\Users\lab07\Documents\pybamm2

C:\Users\lab07\Documents\pybamm2>python -m venv ./venv/

C:\Users\lab07\Documents\pybamm2>.\venv\Scripts\activate.bat

(venv) C:\Users\lab07\Documents\pybamm2>pip install pybamm tqdm matplotlib

(venv) C:\Users\lab07\Documents\pybamm2>python EISdifferentSOC.py
```

and the next time :

```bash
C:\Users\lab07> cd C:\Users\lab07\Documents\pybamm2
C:\Users\lab07\Documents\pybamm2>.\venv\Scripts\activate.bat
(venv) C:\Users\lab07\Documents\pybamm2>python EISdifferentSOC.py
```