# Patient Clustering for Healthcare Efficiency

* This repository holds my attempt apply hierarchical clustering (Agglomeraitve) to Hospital patient discharge data directly from New York's Statewide Planning and Research Cooperative System (SPARCS).

## Overview

  * The task is to use a the N.Y. SPARCS discharge dataset, which contains detailed information on up to 34 patient attributes, as a base to apply a clustering algorithm and provide "data discovery" to better identify groups or "clusters" within the dataset for better organization and clarity of the types of patients.
  * The approach in this repository formulates the problem as a discovery task, comparing patient attributes, gathering relevant information and ultimately running a clustering algorithm on the dataset.
  * The performance of our algorithm is measured by silhouette score:  A measure of how similar something is to its own cluster (also known as its cohesion) versus how similar it is compared to other clusters (also known as its separation). The value goes from −1 to 1, where a value closer to 1 is higher. Our model was able to give us back a silhouette score of *0.19455688449091832*.

## Summary 

### Data

  * Type: 
    * A *.csv* file with 34 patient discharge attributes, specifically pertaining to health and patient condition (ie. Gender, Race, Diagnosis, Age group, Risk of mortality, cost of charges, hospital county, facility ID, etc.) Columns are of integer and object formats.
  * Size: 930 MB
  * Instances: 2.54 million instances for clustering w/ 34 attributes per individual

#### Preprocessing / Clean up
*  Multiple missing/null values for multiple attributes. I used Pandas to remove any null instances that were occuring in the attributes.

*  Some values were not descriptive for our attributes. "Unknown" was a common value found in attributes such as gender, ethnicity, race, etc. Those values were      removed from the dataset as it was deemed irrelevant.

*  Group of attributes that were deemed irrelevant to project including 'Operating Certficiate Number', 'Payment Typology(s)', 'Facility ID', etc. provided no descriptive value to the type of patient that could be discovered, so they were also removed.

*  Attributes listed as "objects" instead of strings, integers, floats, etc. On loadup, these mixed data-type columns were noted to be a problem. Pandas was used to modify columns/attributes into appropriate form.

#### Data Visualization
  
    
![dataframe_head](https://user-images.githubusercontent.com/98781538/235522937-b622b92d-2a8f-4188-848d-9f3a4b6e6593.png)
* Figure 1: A sample showing some of the attributes and patient values
  
    
![LengthofStay](https://user-images.githubusercontent.com/98781538/226444439-c157c899-ae50-4d28-9ee3-54003e7709ad.png)
* Figure 2: One of the most relevant metrics to determine where individuals are clustered into is the amount of time they spent in the hospital. For our dataset, the vast majority of individuals spent 20 days or less as patients.
   
     
![Plotbyage](https://user-images.githubusercontent.com/98781538/226447271-6ca52745-9133-41b4-b430-f0a6cb040633.png)
* Figure 3: This plot shows the frequency of hospitalization increased as age increased, starting from the age of 30 to 70 and upwards. The least hospitalized group (lowest frequency) was 18-29 years of age group. (Note: Group 0-17 has higher hospitalization than 18-29).

  
    
### Problem Formulation

  * Input: Patient instances with varying attributes, specifically pertaining to health and patient condition (ie. Gender, Race, Diagnosis, Age group, Risk of mortality, cost of charges, hospital county, facility ID, etc.). Cleaning and without missing values. Also need to encode categorical values for effective pre-processing to feed into model.
  
  * Output: Groupings of individual patients into larger groups or "Clusters"
  
  * Model(s): 
    * *Agglomerative Clustering*: A type of hierarchical clustering algorithm (An unsupervised technique that divides the dataset into several clusters such that data points in the same cluster are more similar and data points in other clusters are less similar). Agglomerative Clustering uses a bottom-up approach, where every single data point begins as its own cluster and as clusters are merged together the hierarchy is built and the total number of clusters is reduced.
    
    * Metric: *Euclidean Distance*  : 
      
        ![euclideandistance](https://user-images.githubusercontent.com/98781538/235486652-b039b5da-5160-4033-9c2d-cbebcca992e3.png)
         
         *Calculated as the square root of the sum of the squared differences between the two vectors.*
  
  
  
  
  
### Software Used  

* Packages: 

    * Include the regular "stack": NumPy, Matplotlib, Pandas, Sklearn.
    
    * Gower: Package that provides us with the *Gower Distance*: A distance measure that can be used to calculate distance between two entity whose attributes have a mixed of categorical and numerical values. 
          
### Performance

* The performance metric used was Silhoutte score.
  ``` 
  labels = hac.labels_
  silhouette_score(df_scaled_down, labels)
  ```
  ```
  Result: 0.19455688449091832
  ```
     
* Results from *Agglomertive Clustering*:
![dendogram](https://user-images.githubusercontent.com/98781538/235524117-2e7faced-cfff-4d34-8202-63ae1fc01272.png)
  * Figure 4: We create a Dendogram to see how the different clustering occured from our algorithm. This helps build intuition to determine whether or not we are       choosing the appropriate cluster size. Note: Linkage was set to *"average"* 
  
    
![clusters](https://user-images.githubusercontent.com/98781538/235524465-a60444fe-751c-4fd1-83f7-5ee0ad3c16eb.png)
* Figure 5: We incorporate PCA with ```n_components``` set equal to 2, so that we can plot our results.
  
      
  
     
    
### Conclusions

* Given that our Silhouette score was relatively low, there is still room for improvement to tweak the algorithm and/or the data preprocessing. There is also more to be potentially discovered, given that we didn't use the entire data set.
  
* Although our algorithm yielded a interpretable result, the *entire* set of data was not able to be run due to computational issues. This may require other methods/tools if deciding to continue with this algorithm in particular or it is also feasible to switch to another technique if computational tools aren't readily available. 

### Future Work

* Once final result is plotted after performing PCA, attempt to understand specifically why clusters are in certain areas. In future contributions, the investigation of how different metrics affect the clustering could be investigated in greater detail.

* Attempt to formulate problem and define it more efficiently earlier on. Think about the preprocessing so there is less time taken during cleaning. Another note is to consider the libraries and computations I will have to generate, so the work is done at a more readily pace.

* Try other clustering algorithms that may be more computationally viable or attempt full scope of project using entire dataset with more computational tools. Agglomerative clustering doesn't work as well the higher the dataset and complexity are. Other algorithms may be more efficient with given RAM.

## How to reproduce results
 
 * Import neccessary packages:
   ```
   import numpy as np
   import pandas as pd
   import sklearn
   ```

 * Download dataset from hyperlink below. The problem is better approached via Google Colaboratory, so all code is tailored for usage on that particular platform. However, the general approach could also be done on another platform such as Kaggle.
 
 * Use general code provided. However, be aware that file paths will be different depending on where dataset is initially loaded onto Google Colaboratory.

### Files in repository

* *Preproccesing.ipynb*: 
  * Takes *.csv* file via google drive and uploads into dataframe, gets rid of missing values and handles "unknown category" in multiple features.

* *Encoding_and_Visualization.ipynb*: 
  * Notebook that provides code for one-hot encoding along with multiple visualizations/plotting of attributes to help build intuition.

* *Hierarchical_Clustering.ipynb*: 
  * Instantiates the Hierarchical/Agglomerative clustering techniques and runs on data, along with PCA for dimensionality reduction to plot results.

### Software Setup

* Download Gower in jupyter - `!pip install gower`

* Google Colaboratory predominantly used

### Data

* Download link: https://health.data.ny.gov/api/views/u4ud-w55t/rows.csv?accessType=DOWNLOAD&bom=true&format=true

* *.csv* file will have attributes uploaded into Jupyter Notebook as "objects". Optional to change data type option on import and/or adjust values manually.

* Dropped multiple columns via DataFrame manipulation using Pandas functions.


## Citations

* https://medium.com/@rumman1988/clustering-categorical-and-numerical-datatype-using-gower-distance-ab89b3aa90d9

* https://towardsdatascience.com/clustering-on-numerical-and-categorical-features-6e0ebcf1cbad

* [A system for exploring big data: an iterative k-means
searchlight for outlier detection on open health data 
](https://github.com/JBAguinaga/ProjectTemplate/files/11022006/2018.K-means.Health.data.paper.pdf)
