Repo for Lowes project "joint modeling of recommendation and search systems".

### Structure 

``` bash
.
│── preprocess/unified_kgc  # the following 3 files in the folder is for data preprocess.      
│    │─── preprocess.ipynb
│    │─── b25_all.sh
│    └─── create_unified_train.ipynb
│   
│── kgc-dr 
│    │─── trainer/unified_train.py # this file and the following 2 files are for training.        
│    │─── dataset/kgc_triple_dataset.py         
│    │─── modeling/dual_encoder.py
│    │
│    │─── retriever/parallel_index_text_1.py # this file and the following 3 files are for testing.
│    │─── retriever/parallel_index_text_2.py  
│    │─── retriever/retrieve_top_passages.py  
│    │─── evaluation/retriever_evaluator.py  
│    │
│    └─── scripts/unified_kgc # the folder contain the scripts for training and testing.
│            │─── batch_unified_comb_train.sh
│            └─── batch_pipline.sh 
│    
└─── ...
```

### Workflow
- For preprocessing the dataset, run the following files in order (Note that all this files are located in the preprocess/unified_kgc folder):
  ``` preprocess/unified_kgc/preprocess.ipynb --> preprocess/unified_kgc/bm25_all.sh --> preprocess/unified_kgc/create_unified_train.ipynb ```
  After preprocessing, all files for training and testing should be located in the new created foloder: ```datasets/unified_kgc/```
- For training the model, your can go to the folder ```scripts/unified_kgc``` and run the script: ```batch_unified_comb_train.sh```
- After your model trained, you can test your model performance by runing the script: ```batch_unified_comb_train.sh``` which is also located in the folder ```scripts/unified_kgc```. Note that this script, automatically (1) index the collection; (2) retrieve the top documents; (3) compute and ouput the result sequentially. After running the script, you should see the result of the trained model. 