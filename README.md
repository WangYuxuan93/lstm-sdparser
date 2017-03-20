# lstm-parser
Transition based dependency parser with state embeddings computed by LSTM-based Neural Networks (Bi-LSTM Subtraction & Incremental Tree-LSTM)

#### Required software

 * A C++ compiler supporting the [C++11 language standard](https://en.wikipedia.org/wiki/C%2B%2B11)
 * [Boost](http://www.boost.org/) libraries
 * [Eigen](http://eigen.tuxfamily.org) (newer versions strongly recommended)
 * [CMake](http://www.cmake.org/)

#### Checking out the project for the first time

#### Build instructions

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make -j2

#### Train a parsing model

Having a training.conll file and a development.conll formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat), to train a parsing model with the LSTM parser type the following at the command line prompt:

    parser/lstmparse -T training.conll -d development.conll -s swap --data_type text --pretrained_dim 100 --hidden_dim 200 --bilstm_hidden_dim 100 --lstm_input_dim 200 --input_dim 100 --action_dim 50 --pos_dim 50 --rel_dim 50 --dynet_mem 2000 --model_dir models/ --max_itr 40 -P -t
    
Link to the word vectors that we used in the ACL 2015 paper for English:  [sskip.100.vectors](https://drive.google.com/file/d/0B8nESzOdPhLsdWF2S1Ayb1RkTXc/view?usp=sharing).

Note-1: you can also run it without word embeddings by removing the -w option for both training and parsing.

Note-2: the training process should be stopped when the development result does not substantially improve anymore.

Note-3: 

#### Parse data with your parsing model

Having a test.conll file formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat)

    parser/lstmparse -T training.conll -p test.conll -s swap --data_type text --pretrained_dim 100 --hidden_dim 200 --bilstm_hidden_dim 100 --lstm_input_dim 200 --input_dim 100 --action_dim 50 --pos_dim 50 --rel_dim 50 --dynet_mem 2500 -P -m models/parser_swap_pos_nobs_notr_text_2_100_200_50_200_50_50_100-pidXXXX.params

The model name/id is stored where the parser has been trained.
The parser will output the conll file with the parsing result.

#### Pretrained models

TODO

#### Citation

If you make use of this software, please cite the following:

    @inproceedings{dyer:2015acl,
      author={Chris Dyer and Miguel Ballesteros and Wang Ling and Austin Matthews and Noah A. Smith},
      title={Transition-based Dependeny Parsing with Stack Long Short-Term Memory}
      booktitle={Proc. ACL},
      year=2015,
    }

#### License

This software is released under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

#### Contact

For questions and usage issues, please contact yxwang@ir.hit.edu.cn

