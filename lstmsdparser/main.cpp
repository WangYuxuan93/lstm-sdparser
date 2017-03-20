//#include <iostream>

#include <cstring>
#include "lstmsdparser/lstm_sdparser.h"

using lstmsdparser::LSTMParser;
//namespace po = boost::program_options;

cpyp::Corpus corpus;

using namespace dynet::expr;
using namespace dynet;
using namespace std;
namespace po = boost::program_options;

//std::vector<unsigned> possible_actions;
//unordered_map<unsigned, std::vector<float>> pretrained;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
    ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
    ("dev_data,d", po::value<string>(), "Development corpus")
    ("test_data,p", po::value<string>(), "Test corpus")
    ("transition_system,s", po::value<string>()->default_value("list-tree"), "Transition system(swap - arcstandard, list-tree - listbased tree, list-graph - list-graph listbased graph)")
    ("unk_strategy,o", po::value<unsigned>()->default_value(1), "Unknown word strategy: 1 = singletons become UNK with probability unk_prob")
    ("unk_prob,u", po::value<double>()->default_value(0.2), "Probably with which to replace singletons with UNK in training data")
    ("model,m", po::value<string>(), "Load saved model from this file")
    ("use_pos_tags,P", "make POS tags visible to parser")
    ("use_bilstm,B", "use bilstm for buffer")
    ("use_treelstm,R", "use treelstm for subtree in stack")
    ("data_type", po::value<string>()->default_value("sdpv2"), "Data type(sdpv2 - news, text - textbook), only for distinguishing model name")
    ("dynet_seed", po::value<string>(), "Dynet seed for initialization, random initialization if not specified")
    ("dynet_mem", po::value<string>()->default_value("4000"), "Dynet memory size (MB) for initialization")
    ("model_dir", po::value<string>()->default_value(""), "Directory of model")
    ("max_itr", po::value<unsigned>()->default_value(10000), "Max training iteration")
    ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
    ("action_dim", po::value<unsigned>()->default_value(50), "action embedding size")
    ("input_dim", po::value<unsigned>()->default_value(100), "input word embedding size (updated while training)")
    ("hidden_dim", po::value<unsigned>()->default_value(200), "lstm hidden dimension")
    ("bilstm_hidden_dim", po::value<unsigned>()->default_value(100), "bilstm hidden dimension")
    ("pretrained_dim", po::value<unsigned>()->default_value(100), "pretrained input word dimension")
    ("pos_dim", po::value<unsigned>()->default_value(50), "POS dimension")
    ("rel_dim", po::value<unsigned>()->default_value(50), "relation dimension")
    ("lstm_input_dim", po::value<unsigned>()->default_value(200), "LSTM input dimension")
    ("train,t", "Should training be run?")
    ("words,w", po::value<string>()->default_value(""), "Pretrained word embeddings")
    ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0) {
    cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
    exit(1);
  }
}

int main(int argc, char** argv) {
  cerr << "COMMAND:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  
  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  lstmsdparser::Options Opt;
  Opt.USE_POS = conf.count("use_pos_tags");
  Opt.USE_BILSTM = conf.count("use_bilstm");
  Opt.USE_TREELSTM = conf.count("use_treelstm");
  Opt.max_itr = conf["max_itr"].as<unsigned>();
  cerr << "Max training iteration: " << Opt.max_itr << endl;
  if (Opt.USE_BILSTM)
    cerr << "Using bilstm for buffer." << endl;
  if (Opt.USE_TREELSTM)
    cerr << "Using treelstm for subtree in stack." << endl;
  Opt.transition_system = conf["transition_system"].as<string>();
  cerr << "Transition System: " << Opt.transition_system << endl;
  Opt.LAYERS = conf["layers"].as<unsigned>();
  Opt.INPUT_DIM = conf["input_dim"].as<unsigned>();
  Opt.PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
  Opt.HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
  Opt.ACTION_DIM = conf["action_dim"].as<unsigned>();
  Opt.LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
  Opt.POS_DIM = conf["pos_dim"].as<unsigned>();
  Opt.REL_DIM = conf["rel_dim"].as<unsigned>();
  Opt.BILSTM_HIDDEN_DIM = conf["bilstm_hidden_dim"].as<unsigned>(); // [bilstm]
  if (conf.count("dynet_seed")){
    Opt.dynet_seed = conf["dynet_seed"].as<string>();
  }
  if (conf.count("dynet_mem")){
    Opt.dynet_mem = conf["dynet_mem"].as<string>();
  }
  const unsigned unk_strategy = conf["unk_strategy"].as<unsigned>();
  cerr << "Unknown word strategy: ";
  if (unk_strategy == 1) {
    cerr << "STOCHASTIC REPLACEMENT\n";
  } else {
    abort();
  }
  const double unk_prob = conf["unk_prob"].as<double>();
  assert(unk_prob >= 0.); assert(unk_prob <= 1.);
  ostringstream os;
  os << conf["model_dir"].as<string>()
    << "parser_" << Opt.transition_system
    << '_' << (Opt.USE_POS ? "pos" : "nopos")
    << '_' << (Opt.USE_BILSTM ? "bs" : "nobs")
    << '_' << (Opt.USE_TREELSTM ? "tr" : "notr")
    << '_' << conf["data_type"].as<string>()
    << '_' << Opt.LAYERS
    << '_' << Opt.INPUT_DIM
    << '_' << Opt.HIDDEN_DIM
    << '_' << Opt.ACTION_DIM
    << '_' << Opt.LSTM_INPUT_DIM
    << '_' << Opt.POS_DIM
    << '_' << Opt.REL_DIM
    << '_' << Opt.BILSTM_HIDDEN_DIM
    << "-pid" << getpid() << ".params";

  const string fname = os.str();
  cerr << "Writing parameters to file: " << fname << endl;

  LSTMParser *parser = new LSTMParser();
  //parser -> set_options(Opt);

  parser->DEBUG = true;
  if (conf.count("model") && conf.count("dev_data")){
    parser -> set_options(Opt); // only for test
    parser -> load(conf["model"].as<string>(), conf["training_data"].as<string>(), 
                  conf["words"].as<string>(), conf["dev_data"].as<string>() );
  }
  else if (conf.count("model") && conf.count("test_data")){
    parser -> set_options(Opt); // only for test
    parser -> load(conf["model"].as<string>(), conf["training_data"].as<string>(), 
                  conf["words"].as<string>());
  }
  else{
    parser -> set_options(Opt);
    parser -> load("", conf["training_data"].as<string>(), 
                  conf["words"].as<string>(), conf["dev_data"].as<string>() );
  }

  // OOV words will be replaced by UNK tokens
  //TRAINING
  
  if (conf.count("train")) {
    parser->train(fname, unk_strategy, unk_prob);
  } // should do training?
  
  if (conf.count("dev_data")) { // do test evaluation
    parser->predict_dev();
  }
  if (conf.count("test_data")){
    parser->test(conf["test_data"].as<string>());
  }
}

