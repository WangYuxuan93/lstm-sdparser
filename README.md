# lstm-parser
Transition based dependency parser with state embeddings computed by LSTM-based Neural Networks (Bi-LSTM Subtraction & Incremental Tree-LSTM)

#### Required software

 * A C++ compiler supporting the [C++11 language standard](https://en.wikipedia.org/wiki/C%2B%2B11)
 * [Boost](http://www.boost.org/) libraries
 * [Eigen](https://bitbucket.org/eigen/eigen) (Development version)
 * [CMake](http://www.cmake.org/)

If you don't have Eigen already, you can get it easily using the following command:


    hg clone https://bitbucket.org/eigen/eigen/ -r 699b659


#### Checking out the project for the first time

#### Build instructions

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make -j2

#### Train a parsing model

Having a training.conll file and a development.conll formatted according to the [SemEval16 Task9 data format](http://alt.qcri.org/semeval2016/task9/index.php?id=data-and-tools), to train a parsing model with the LSTM parser type the following at the command line prompt:

    parser/lstmparse -T training.conll -d development.conll -s list-graph --data_type text --pretrained_dim 100 --hidden_dim 200 --bilstm_hidden_dim 100 --lstm_input_dim 200 --input_dim 100 --action_dim 50 --pos_dim 50 --rel_dim 50 --dynet_mem 2000 --model_dir models/ --max_itr 5000 -P -t

Note-1: You can also run it without word embeddings by removing the -w option for both training and parsing.

Note-2: The training process should be stopped when the development result does not substantially improve anymore.

Note-3: The transition system (specified by -s option) includes "swap", "list-tree" and "list-graph", the previous two only parse dependency trees, the latter parse dependency graphs. 

Note-4: You can train the parser with Bi-LSTM Subtraction (-B) and Incremental Tree-LSTM (-R) with the corresponding option. Use -h option to get specific descriptions of the options.

#### Parse data with your parsing model

Having a test.conll file formatted according to the [SemEval16 Task9 data format](http://alt.qcri.org/semeval2016/task9/index.php?id=data-and-tools)

    parser/lstmparse -T training.conll -p test.conll -s list-graph --data_type text --pretrained_dim 100 --hidden_dim 200 --bilstm_hidden_dim 100 --lstm_input_dim 200 --input_dim 100 --action_dim 50 --pos_dim 50 --rel_dim 50 --dynet_mem 2500 -P -m models/parser_list-graph_pos_nobs_notr_text_2_100_200_50_200_50_50_100-pidXXXX.params

The parser will output the conll file with the parsing result.

#### Pretrained models

TODO

#### Citation

If you make use of this software, please cite the following:

    @inproceedings{wang:2018aaai,
      author={Wang Yuxuan and Che Wanxiang and Guo Jiang and Liu Ting},
      title={A Neural Transition-Based Approach for Semantic Dependency Graph Parsing}
      booktitle={Proc. AAAI},
      year=2018,
    }

#### License

This software is released under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

#### Contact

For questions and usage issues, please contact yxwang@ir.hit.edu.cn

