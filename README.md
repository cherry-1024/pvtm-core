<<<<<<< HEAD
# Paragraph Vector Topic Model



This project providers users the ability to do topic modeling with [Paragraph Vector Topic Models](https://www.uni-marburg.de/fb02/makro/forschung/magkspapers/paper_2018/15-2018_lenz.pdf).

<img src="img/topic.png" height="300px" width="375px" align="left">
<img src="img/timeline.png" height="300px" width="375px" >



## Installation

Pull the repo from github.
```
git clone https://github.com/davidlenz/pvtm-core
cd pvtm-core
```


Install the required packages using pip
```
pip install -r requirements.txt
```


To verify correct setup, run the following:
```
python pvtm/pvtm.py -i data/sample_100.csv
```

## Usage

Run
```
python pvtm/pvtm.py -help
```
to see options.


## Visualizations
Once we have created an output folder with outputs from pvtm by running the above command, we can visualize the results using
```
python pvtm/pvtm_vis.py -path <path-to-folder>
```

Note that the visualizations require an R installation which also needs to be in your system path.
=======
# Paragraph Vector Topic Model



This project providers users the ability to do topic modeling with [Paragraph Vector Topic Models](https://www.uni-marburg.de/fb02/makro/forschung/magkspapers/paper_2018/15-2018_lenz.pdf).

<img src="img/topic.png" height="300px" width="375px" align="left">
<img src="img/timeline.png" height="300px" width="375px" >



## Installation

Pull the repo from github.
```
git clone https://github.com/davidlenz/pvtm-core
cd pvtm-core
```


Install the required packages using pip
```
pip install -r requirements.txt
```


To verify correct setup, run the following:
```
python pvtm/pvtm.py -i data/sample_100.csv
```

## Usage

Run
```
python pvtm/pvtm.py -help
```
to see options.


## Visualizations
Once we have created an output folder with outputs from pvtm by running the above command, we can visualize the results using
```
python pvtm/pvtm_vis.py -path <path-to-folder>
```

Note that the visualizations require an R installation which also needs to be in your system path.
>>>>>>> 588749c34b8acf10dc00b48d3e82173308ce475d
