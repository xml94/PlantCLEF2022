
## Reproduce our results for PlantCLEF2022 challenge
* Make datasets, look and adapt ```make_dataset.sh```, and then run the command.
* Visualize the dataset, use ```visualize_observation.py```.
* Download MAE pretrained model [ViT-Large](https://github.com/facebookresearch/mae) and put it in ```checkpoint``` directory.
* Finetune the model in PlantCLEF2022 dataset, use ```finetune.sh```.
* Test and predict the results, use ```test.sh```, you can see a new directory in ```results``` directory.
* Make the submission result, use ```make_official_result.sh``` you can see a new ```.csv``` file in ```results``` directory.


## Our results
| Name                                   | MA-MRR  | Our model                                                                                                   |
|----------------------------------------|---------|-------------------------------------------------------------------------------------------------------------|
| official run 8: epoch 80 Single_high   | 0.62692 | No                                                                                                          | 
| late submission epoch 100 Single_high  | 0.63668 | [Google Driver](https://drive.google.com/drive/folders/1JCVX58oVZFuIttPHaeAjs_zkMXkzzJeA?usp=sharing)       |
| late submission epoch 100 multi_sorted | 0.64079 | [same with last line](https://drive.google.com/drive/folders/1JCVX58oVZFuIttPHaeAjs_zkMXkzzJeA?usp=sharing) |


## Citation
```
@inproceedings{xu2022transfer,
  title={Transfer learning with self-supervised vision transformer for large-scale plant identification},
  author={Xu, Mingle and Yoon, Sook and Jeong, Yongchae and Lee, Jaesu and Park, Dong Sun},
  booktitle={International conference of the cross-language evaluation forum for European languages (Springer;)},
  pages={2253--2261},
  year={2022}
}
@article{xu2022transfer,
  title={Transfer learning for versatile plant disease recognition with limited data},
  author={Xu, Mingle and Yoon, Sook and Jeong, Yongchae and Park, Dong Sun},
  journal={Frontiers in Plant Science},
  volume={13},
  pages={4506},
  year={2022},
  publisher={Frontiers}
}
```

## Acknowledgement
Our model is heavily based on [MAE](https://github.com/facebookresearch/mae).
