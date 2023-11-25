# Football analytics of Serie A

This folder contains scripts to run the Bayesian ranking system developed by Microsoft during the Serie A season 2018/19.
## Dependencies

Create a new virtual environment and install all the necessary Python packages:

```
python3 -m venv sports-analytics-env
source sports-analytics-env/bin/activate
pip install --upgrade pip
python3 -m pip install -r sports-analytics-env/requirements.txt
```

## Content

Run `gibbs_season.py` to generate the plots.