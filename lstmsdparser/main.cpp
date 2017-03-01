//#include <iostream>

#include <cstring>
#include "lstmsdparser/lstm_sdparser.h"

using ltp::lstmsdparser::LSTMParser;
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
        ("transition_system,s", po::value<string>()->default_value("list"), "Transition system(arcstd - arcstandard, list - listbased, tree - tree listbased)")
        ("data_type,k", po::value<string>()->default_value("sdpv2"), "Data type(sdpv2 - news, text - textbook)")
        ("dynet_seed", po::value<string>(), "Dynet seed for initialization")
        ("dynet_mem", po::value<string>()->default_value("4000"), "Dynet memory size (MB) for initialization")
        ("unk_strategy,o", po::value<unsigned>()->default_value(1), "Unknown word strategy: 1 = singletons become UNK with probability unk_prob")
        ("unk_prob,u", po::value<double>()->default_value(0.2), "Probably with which to replace singletons with UNK in training data")
        ("model,m", po::value<string>(), "Load saved model from this file")
        ("use_pos_tags,P", "make POS tags visible to parser")
        ("use_bilstm,B", "use bilstm for buffer")
        ("use_treelstm,R", "use treelstm for subtree in stack")
        ("beam_size,b", po::value<unsigned>()->default_value(0), "beam size")
        ("global_loss,G", "train using the global loss function (Andor et al)")
        ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
        ("action_dim", po::value<unsigned>()->default_value(50), "action embedding size")
        ("input_dim", po::value<unsigned>()->default_value(100), "input embedding size")
        ("hidden_dim", po::value<unsigned>()->default_value(200), "hidden dimension")
        ("bilstm_hidden_dim", po::value<unsigned>()->default_value(100), "bilstm hidden dimension")
        ("pretrained_dim", po::value<unsigned>()->default_value(100), "pretrained input dimension")
        ("pos_dim", po::value<unsigned>()->default_value(50), "POS dimension")
        ("rel_dim", po::value<unsigned>()->default_value(50), "relation dimension")
        ("lstm_input_dim", po::value<unsigned>()->default_value(200), "LSTM input dimension")
        ("train,t", "Should training be run?")
        ("words,w", po::value<string>(), "Pretrained word embeddings")
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
  //dynet::Initialize(argc, argv);

  cerr << "COMMAND:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  
  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  //ltp::lstmsdparser::Sizes System_size;
  ltp::lstmsdparser::Options Opt;
  Opt.USE_POS = conf.count("use_pos_tags");
  Opt.USE_BILSTM = conf.count("use_bilstm");
  Opt.USE_TREELSTM = conf.count("use_treelstm");
  Opt.GLOBAL_LOSS = conf.count("global_loss");
  Opt.beam_size = conf["beam_size"].as<unsigned>();
  if (Opt.USE_BILSTM)
    cerr << "Using bilstm for buffer." << endl;
  if (Opt.USE_TREELSTM)
    cerr << "Using treelstm for subtree in stack." << endl;
  if (Opt.beam_size > 0)
    cerr << "Using beam search, beam size: " << Opt.beam_size << endl;

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
  os << "parser_" << Opt.transition_system
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

  //Model model;
  LSTMParser *parser = new LSTMParser();
  //parser -> set_options(Opt);

  parser->DEBUG = true;
  if (conf.count("model")){
    parser -> set_options(Opt); // only for test
    parser -> load(conf["model"].as<string>(), conf["training_data"].as<string>(), 
                  conf["words"].as<string>(), conf["dev_data"].as<string>() );
  }
  else{
    parser -> set_options(Opt);
    parser -> load("", conf["training_data"].as<string>(), 
                  conf["words"].as<string>(), conf["dev_data"].as<string>() );
  }
  //parser -> get_dynamic_infos();
  /*
  if (conf.count("model")) {
    ifstream in(conf["model"].as<string>().c_str());
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }*/

  // OOV words will be replaced by UNK tokens
  
  //TRAINING
  
  if (conf.count("train")) {
    parser->train(fname, unk_strategy, unk_prob);
  } // should do training?
  
  if (true) { // do test evaluation
    parser->predict_dev(); // id : 22 146 296 114 21
    /*
    std::vector<std::vector<string>> hyp;
    string word[]={"我","是","中国","学生","ROOT"}; // id : 22 146 296 114 21
    size_t w_count=sizeof(word)/sizeof(string);
    string pos[]={"NN","VE","JJ","NN","ROOT"};
    size_t p_count=sizeof(word)/sizeof(string);
    std::vector<std::string> words(word,word+w_count);
    std::vector<std::string> postags(pos,pos+p_count);
    parser -> predict(hyp, words, postags);
    for (int i = 0; i < hyp.size(); i++){
      for (int j = 0; j < hyp.size(); j++)
        cerr << hyp[i][j] << " ";
      cerr << endl;
    }*/
  }
  //for (unsigned i = 0; i < corpus.actions.size(); ++i) {
    //cerr << corpus.actions[i] << '\t' << parser.p_r->values[i].transpose() << endl;
    //cerr << corpus.actions[i] << '\t' << parser.p_p2a->values.col(i).transpose() << endl;
  //}
}

