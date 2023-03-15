![Alt text](./images/banner.png?raw=true "Banner")

# :evergreen_tree: Data sustainability challenge
## Hydrogen station roll out roadmap

The goal of this project is to model the rollout of a hydrogen station plan using several different scenarios. The challenge was broken into 4 distinct parts to address the different challenges of rolling out a hydrogen charging network in France

Part 1: Size the network of H2 truck charging stations in France in 2030 & 2040 (number of stations and repartition per region), depending on:
  * The forecasted number of H2 trucks in France and Europe
  * The autonomy of trucks
  * The driver regulations
  * The motorway network in France

Part 2: Define models to identify the exact locations where to implement H2 stations, depending on:
  * The truck traffic per transit axis
  * The localization of logistic hubs
  * The cost (deployment and operations) per station

Part 3 : Apply the models to France and build a 2030 & 2040 deployment roadmap, depending on three competitive scenarios:
  * 1: Only one network in France
  * 2: Two players entering simultaneously the market
  * 3: One player entering after an incumbent (transforming its oil stations network to H2)

Part 4: Localization of the H2 production infrastructure, depending on:
  * The H2 production costs
  * The H2 transport costs

## 🚀 Getting started with the repository

To ensure that all libraries are installed pip install the requirements file:
 
```
pip install -r requirements.txt
```

To run the webapp go to the console and run following command: 
 
```
cd app
streamlit run Home.py
```

You should be at the source of the repository structure (ie. sust_challenge) when running the command.

## 🗂 Repository structure

Our repository is structured in the following way:
```
|sust_challenge
   |--app
   |-----.streamlit
   |-----pages
   |-----Home.py
   |--competitive_analysis
   |-----competitive_analysis.py
   |--images
   |--load_preprocess
   |-----functions.py
   |-----predictions.py
   |--params
   |------config.json
   |--station_finder
   |-----functions.py
   |--notebook.ipynb
   |--README.md
   |--requirements.txt
```
# File overview
## :apple: App

This folder contains all the web app data and the corresponding pages. There are 5 seperate pages and a Home.py function to run the webapp from.

## :1st_place_medal: Competitive analysis
This folder contains the competitive analysis for the four different scenarios described in part 1 of the challenge. We define a baseline, worst case, best case, and truck-type sensitivity scenario as our output. It is also used to derive information from the competitive scenarios in part 3.

## :floppy_disk: Load Preprocess
This folder contains the function to load all the necessery data and derive analyses for part 1 of the challenge.

## :abacus: Params
This folder contains all the parameters that we define before running our functions. These parameters are configurable.

## :eyeglasses: Station Finder
This folder contains our functions used for parts 2, 3, and 4. The StationLocator class is used to determine the optimal locations for part 2. The Scenarios and Case class are used to determine the best locations for our stations based on our competitors. The ProductionLocator is used in part 4 to determine the optimal location of our production sites based on the location of our hydrogen stations that we defined in part 3.

# :mailbox_with_no_mail: Linkedin Contact

If you have any feedback, please reach out to us on LinkedIn!
- [Lea Chader](https://www.linkedin.com/in/lea-chader/)
- [Inès Benito](https://www.linkedin.com/in/ines-benito/)
- [Adrian (Kun) Tan](https://www.linkedin.com/in/kun-tan/)
- [Salah Mahmoudi](https://www.linkedin.com/in/salahmahmoudi/)
- [Bechara Fayad](https://lb.linkedin.com/in/bechara-fayad-8a83b6114)



