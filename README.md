## Learning to Separate Object Sounds by Watching Unlabeled Video
Learning to Separate Object Sounds by Watching Unlabeled Video: [[Project Page]](http://vision.cs.utexas.edu/projects/separating_object_sounds/)    [[arXiv]](https://arxiv.org/pdf/1804.01665.pdf)<br/>

This repository contains the deep MIML network implementation for our [ECCV 2018 paper](http://www.cs.utexas.edu/~grauman/papers/sound-sep-eccv2018.pdf).

If you find our code or project useful in your research, please cite:

        @inproceedings{gao2018objectSounds,
          title={Learning to Separate Object Sounds by Watching Unlabeled Video},
          author={Gao, Ruohan and Feris, Rogerio and Grauman, Kristen},
          booktitle={ECCV},
          year={2018}
        }
     
Use the following command to train the deep MIML network:
  ```
  python train.py --HDF5FileRoot /your_hdf5_file_root --name deepMIML --checkpoints_dir checkpoints --model MIML --batchSize 256 --learning_rate 0.001 --learning_rate_decrease_itr 5 --decay_factor 0.94 --display_freq 10 --save_epoch_freq 5 --save_latest_freq 500 --gpu_ids 0 --nThreads 2 --num_of_fc 1 --with_batchnorm --continue_train --niter 300 --L 15 --validation_on --validation_freq 50 --validation_batches 10 --selected_classes --using_multi_labels |& tee -a train.log
  ```
  
### Acknowlegements
Our code borrows heavily from the the CycleGAN implementation https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/.
