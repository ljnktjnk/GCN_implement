# Implement GCN for key extract information task

## Create Dataset

1. Create "./GCN_data" folder

2. In "./GCN_data", create "csv" and "images" folders
    - file "csv" contain infors: "xmin\tymin\txmax\tymax\tobject\tlabel".
    - object is text of box
    - label is name of entity
    - ex:   
        xmin    ymin    xmax    ymax    object  label
        0   0   367.3506654366668   44.247160809461434  WORK EXPERIENCE 
        0   44.247160809461434  267.33048928901474  97.03246232944662   ObblaTools, Inc.    com_name
        1198.6304715600907  52.58168210209067   1459.237588381868   94.25428856523706   June 2020 - Present time

3. Create graph dataset for train and test
    python dataset.py -input {csv folder contain csv file} -output {folder contain train and test dataset} -cls {list of labels} 

## Train
    python train.py 

# Test
    python test.py
