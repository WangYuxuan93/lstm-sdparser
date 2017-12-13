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

#### Examples

Examples of training and predicting shell are in examples file, you can try them by

    cd examples
    ./train.sh
    ./test.sh

#### Data

The data should be formatted according to [SemEval16 Task9 data format](http://alt.qcri.org/semeval2016/task9/index.php?id=data-and-tools). Here is what it is look like:

    1	早起	早起	AD	AD	_	2	Exp	_	_
    2	使	使	VV	VV	_	0	Root	_	_
    3	人	人	NN	NN	_	2	Datv	_	_
    3	人	人	NN	NN	_	4	Exp	_	_
    4	健康	健康	VV	VV	_	2	eResu	_	_

Note that the multi-head words are represented by multiple lines.

#### Train a parsing model

Having a training.conll file and a development.conll formatted according to the [SemEval16 Task9 data format](http://alt.qcri.org/semeval2016/task9/index.php?id=data-and-tools), to train a parsing model with the LSTM parser type the following at the command line prompt:

    ../build/lstmsdparser/lstmparse -T data/trail.train.conll -d data/trail.dev.conll -w data/trail.emb -s list-graph --data_type trail --pretrained_dim 100 --hidden_dim 200 --bilstm_hidden_dim 100 --lstm_input_dim 200 --input_dim 100 --action_dim 50 --pos_dim 50 --rel_dim 50 --dynet_mem 2000 --max_itr 5000 -c sem16 -P -B -R -t --model_dir models/

Note-1: You can also run it without word embeddings by removing the -w option for both training and parsing.

Note-2: The training process should be stopped when the development result does not substantially improve anymore.

Note-3: The transition system (specified by -s option) includes "swap", "list-tree" and "list-graph", the previous two only parse dependency trees, the latter parse dependency graphs. 

Note-4: You can train the parser with Bi-LSTM Subtraction (-B) and Incremental Tree-LSTM (-R) with the corresponding option. Use -h option to get specific descriptions of the options.

Note-5: Use -c to specify the type of corpus (sem15 or sem16) you are using. While training on Sem15, remenber to include --has_head option.

Note-6: The Sem15 corpus should be preprocessed according to the paper, thus the input format should be the same as shown above.

#### Parse data with your parsing model

Having a test.conll file formatted according to the [SemEval16 Task9 data format](http://alt.qcri.org/semeval2016/task9/index.php?id=data-and-tools)

    ../build/lstmsdparser/lstmparse -p data/trail.test.conll -s list-graph --data_type trail --pretrained_dim 100 --hidden_dim 200 --bilstm_hidden_dim 100 --lstm_input_dim 200 --input_dim 100 --action_dim 50 --pos_dim 50 --rel_dim 50 --dynet_mem 2000 --max_itr 5000 -c sem16 -P -B -R -m models/trail.model > trail.pred.conll

Note that the options -B and -R should be exactly the same as specified when training the model you are loading.
 
The parser will output the conll file with the parsing result.

Note:  Use --sdp_output option while predicting on Sem15 to directly produce the Sem15 format.

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

