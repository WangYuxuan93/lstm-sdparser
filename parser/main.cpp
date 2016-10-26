//#include <iostream>

#include <cstring>
#include "lstm_sdparser.h"

using ltp::lstmsdparser::LSTMParser;
//namespace po = boost::program_options;

cpyp::Corpus corpus;
volatile bool requested_stop = false;

using namespace cnn::expr;
using namespace cnn;
using namespace std;
namespace po = boost::program_options;

std::vector<unsigned> possible_actions;
unordered_map<unsigned, std::vector<float>> pretrained;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
        ("dev_data,d", po::value<string>(), "Development corpus")
        ("test_data,p", po::value<string>(), "Test corpus")
        ("transition_system,s", po::value<string>()->default_value("list"), "Transition system(list - listbased, spl - simplified)")
        ("data_type,k", po::value<string>()->default_value("sdpv2"), "Data type(sdpv2 - news, text - textbook)")
        ("unk_strategy,o", po::value<unsigned>()->default_value(1), "Unknown word strategy: 1 = singletons become UNK with probability unk_prob")
        ("unk_prob,u", po::value<double>()->default_value(0.2), "Probably with which to replace singletons with UNK in training data")
        ("model,m", po::value<string>(), "Load saved model from this file")
        ("use_pos_tags,P", "make POS tags visible to parser")
        ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
        ("action_dim", po::value<unsigned>()->default_value(16), "action embedding size")
        ("input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
        ("hidden_dim", po::value<unsigned>()->default_value(64), "hidden dimension")
        ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
        ("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
        ("rel_dim", po::value<unsigned>()->default_value(10), "relation dimension")
        ("lstm_input_dim", po::value<unsigned>()->default_value(60), "LSTM input dimension")
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

void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}

void output_conll(const vector<unsigned>& sentence, const vector<unsigned>& pos,
                  const vector<string>& sentenceUnkStrings, 
                  const map<unsigned, string>& intToWords, 
                  const map<unsigned, string>& intToPos, 
                  const vector<vector<string>>& hyp) {
    for (unsigned i = 0; i < (sentence.size()-1); ++i) {
        auto index = i + 1;
        assert(i < sentenceUnkStrings.size() && 
            ((sentence[i] == corpus.get_or_add_word(cpyp::Corpus::UNK) &&
                sentenceUnkStrings[i].size() > 0) ||
                (sentence[i] != corpus.get_or_add_word(cpyp::Corpus::UNK) &&
                sentenceUnkStrings[i].size() == 0 &&
                intToWords.find(sentence[i]) != intToWords.end())));
        string wit = (sentenceUnkStrings[i].size() > 0)? 
        sentenceUnkStrings[i] : intToWords.find(sentence[i])->second;
        auto pit = intToPos.find(pos[i]);
        for (unsigned j = 0; j < sentence.size() ; ++j){
            if (hyp[j][i] != ltp::lstmsdparser::REL_NULL){
                auto hyp_head = j + 1;
                if (hyp_head == sentence.size()) hyp_head = 0;
                auto hyp_rel = hyp[j][i];
                cout << index << '\t'       // 1. ID 
                    << wit << '\t'         // 2. FORM
                    << "_" << '\t'         // 3. LEMMA 
                    << "_" << '\t'         // 4. CPOSTAG 
                    << pit->second << '\t' // 5. POSTAG
                    << "_" << '\t'         // 6. FEATS
                    << hyp_head << '\t'    // 7. HEAD
                    << hyp_rel << '\t'     // 8. DEPREL
                    << "_" << '\t'         // 9. PHEAD
                    << "_" << endl;        // 10. PDEPREL
            }
        }
  }
  cout << endl;
}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  cerr << "COMMAND:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 100;

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  ltp::lstmsdparser::Sizes System_size;
  ltp::lstmsdparser::Options Opt;
  Opt.USE_POS = conf.count("use_pos_tags");

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
  os << "parser_" << (Opt.USE_POS ? "pos" : "nopos")
     << '_' << conf["data_type"].as<string>()
     << '_' << Opt.LAYERS
     << '_' << Opt.INPUT_DIM
     << '_' << Opt.HIDDEN_DIM
     << '_' << Opt.ACTION_DIM
     << '_' << Opt.LSTM_INPUT_DIM
     << '_' << Opt.POS_DIM
     << '_' << Opt.REL_DIM
     << "-pid" << getpid() << ".params";
  int best_LF = 0;
  const string fname = os.str();
  cerr << "Writing parameters to file: " << fname << endl;
  bool softlinkCreated = false;
  corpus.load_correct_actions(conf["training_data"].as<string>());	
  const unsigned kUNK = corpus.get_or_add_word(cpyp::Corpus::UNK);
  System_size.kROOT_SYMBOL = corpus.get_or_add_word(ltp::lstmsdparser::ROOT_SYMBOL);

  if (conf.count("words")) {
    pretrained[kUNK] = std::vector<float>(Opt.PRETRAINED_DIM, 0);
    cerr << "Loading from " << conf["words"].as<string>() << " with" << Opt.PRETRAINED_DIM << " dimensions\n";
    ifstream in(conf["words"].as<string>().c_str());
    string line;
    getline(in, line);
    std::vector<float> v(Opt.PRETRAINED_DIM, 0);
    string word;
    while (getline(in, line)) {
      istringstream lin(line);
      lin >> word;
      for (unsigned i = 0; i < Opt.PRETRAINED_DIM; ++i) lin >> v[i];
      unsigned id = corpus.get_or_add_word(word);
      pretrained[id] = v;
    }
  }

  set<unsigned> training_vocab; // words available in the training corpus
  set<unsigned> singletons;
  {  // compute the singletons in the parser's training data
    map<unsigned, unsigned> counts;
    for (auto sent : corpus.sentences)
      for (auto word : sent.second) { training_vocab.insert(word); counts[word]++; }
    for (auto wc : counts)
      if (wc.second == 1) singletons.insert(wc.first);
  }

  cerr << "Number of words: " << corpus.nwords << endl;
  System_size.VOCAB_SIZE = corpus.nwords + 1;
  //ACTION_SIZE = corpus.nactions + 1;
  System_size.ACTION_SIZE = corpus.nactions + 30; // leave places for new actions in test set
  System_size.POS_SIZE = corpus.npos + 10;  // bad way of dealing with the fact that we may see new POS tags in the test set
  possible_actions.resize(corpus.nactions);
  for (unsigned i = 0; i < corpus.nactions; ++i)
    possible_actions[i] = i;

  Model model;
  LSTMParser *parser = new LSTMParser();
  parser -> set_options(Opt);
  parser -> load(conf["model"].as<string>(), pretrained, possible_actions, System_size);
  parser -> setup_system();
  /*
  if (conf.count("model")) {
    ifstream in(conf["model"].as<string>().c_str());
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }*/

  // OOV words will be replaced by UNK tokens
  corpus.load_correct_actionsDev(conf["dev_data"].as<string>());
  //TRAINING
  if (conf.count("train")) {
    signal(SIGINT, signal_callback_handler);
    SimpleSGDTrainer sgd(&model);
    //MomentumSGDTrainer sgd(&model);
    sgd.eta_decay = 0.08;
    //sgd.eta_decay = 0.05;
    std::vector<unsigned> order(corpus.nsentences);
    for (unsigned i = 0; i < corpus.nsentences; ++i)
      order[i] = i;
    double tot_seen = 0;
    status_every_i_iterations = min(status_every_i_iterations, corpus.nsentences);
    unsigned si = corpus.nsentences;
    cerr << "NUMBER OF TRAINING SENTENCES: " << corpus.nsentences << endl;
    unsigned trs = 0;
    double right = 0;
    double llh = 0;
    bool first = true;
    int iter = -1;
    //time_t time_start = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    time_t time_start = time(NULL);
    std::string t_s(asctime(localtime(&time_start))); 
    //cerr << "TRAINING STARTED AT: " << asctime(localtime(&time_start)) << endl;
    cerr << "TRAINING STARTED AT: " << t_s.substr(0, t_s.size() - 1) << endl;
    while(!requested_stop) {
      ++iter;
      for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
           if (si == corpus.nsentences) {
             si = 0;
             if (first) { first = false; } else { sgd.update_epoch(); }
             cerr << "**SHUFFLE\n";
             random_shuffle(order.begin(), order.end());
           }
           tot_seen += 1;
           const std::vector<unsigned>& sentence=corpus.sentences[order[si]];
           std::vector<unsigned> tsentence=sentence;
           if (unk_strategy == 1) {
             for (auto& w : tsentence)
               if (singletons.count(w) && cnn::rand01() < unk_prob) w = kUNK;
           }
	   const std::vector<unsigned>& sentencePos=corpus.sentencesPos[order[si]]; 
	   const std::vector<unsigned>& actions=corpus.correct_act_sent[order[si]];
           ComputationGraph hg;
           //cerr << "Start word:" << corpus.intToWords[sentence[0]]<<corpus.intToWords[sentence[1]] << endl;
           std::vector<std::vector<string>> cand;
           parser->log_prob_parser(&hg,sentence,tsentence,sentencePos,actions,corpus.actions,corpus.intToWords,&right,cand);
           double lp = as_scalar(hg.incremental_forward());
           if (lp < 0) {
             cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp << endl;
             assert(lp >= 0.0);
           }
           hg.backward();
           sgd.update(1.0);
           llh += lp;
           ++si;
           trs += actions.size();
      }
      sgd.status();
      //time_t time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
      time_t time_now = time(NULL);
      std::string t_n(asctime(localtime(&time_now))); 
      cerr << "update #" << iter << " (epoch " << (tot_seen / corpus.nsentences) << " |time=" << t_n.substr(0, t_n.size() - 1) << ")\tllh: "<< llh<<" ppl: " << exp(llh / trs) << " err: " << (trs - right) / trs << endl;
      llh = trs = right = 0;

      static int logc = 0;
      ++logc;
      if (logc % 25 == 1) { // report on dev set
        unsigned dev_size = corpus.nsentencesDev;
        // dev_size = 100;
        double llh = 0;
        double trs = 0;
        double right = 0;
        //double correct_heads = 0;
        //double total_heads = 0;
        auto t_start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<std::vector<string>>> refs, hyps;
        for (unsigned sii = 0; sii < dev_size; ++sii) {
           const std::vector<unsigned>& sentence=corpus.sentencesDev[sii];
	   const std::vector<unsigned>& sentencePos=corpus.sentencesPosDev[sii]; 
	   const std::vector<unsigned>& actions=corpus.correct_act_sentDev[sii];
           std::vector<unsigned> tsentence=sentence;
           for (auto& w : tsentence)
             if (training_vocab.count(w) == 0) w = kUNK;

           ComputationGraph hg;
           std::vector<std::vector<string>> cand;
            std::vector<unsigned> pred = parser->log_prob_parser(&hg,sentence,tsentence,sentencePos,std::vector<unsigned>(),
                                                                                                corpus.actions,corpus.intToWords,&right,cand);
           double lp = 0;
           llh -= lp;
           trs += actions.size();
           //cerr << "start word:" << sii << corpus.intToWords[sentence[0]] << corpus.intToWords[sentence[1]] << endl;
           std::vector<std::vector<string>> ref = parser->compute_heads(sentence, actions, corpus.actions);
           std::vector<std::vector<string>> hyp = parser->compute_heads(sentence, pred, corpus.actions);
           //output_conll(sentence, corpus.intToWords, ref, hyp);
           //correct_heads += compute_correct(ref, hyp, sentence.size() - 1);
           //total_heads += sentence.size() - 1;
           refs.push_back(ref);
           hyps.push_back(hyp);
        }
        map<string, double> results = ltp::lstmsdparser::evaluate(refs, hyps, corpus.sentencesPosDev, corpus.posToInt["PU"]);
        auto t_end = std::chrono::high_resolution_clock::now();
        cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.nsentences) << ")\tllh=" << llh 
                << " ppl: " << exp(llh / trs) << " err: " << (trs - right) / trs << " LF: " << results["LF"] << " UF:" << results["UF"] 
                << " LP:" << results["LP"] << " LR:" << results["LR"] << " UP:" << results["UP"] << " UR:" <<results["UR"]
                << "\t[" << dev_size << " sents in " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms]" << endl;
        if (results["LF"] > best_LF) {
          cerr << "---saving model to " << fname << "---" << endl;
          best_LF = results["LF"];
          ofstream out(fname);
          boost::archive::text_oarchive oa(out);
          oa << model;
          // Create a soft link to the most recent model in order to make it
          // easier to refer to it in a shell script.
          if (!softlinkCreated) {
            string softlink = " latest_model";
            if (system((string("rm -f ") + softlink).c_str()) == 0 && 
                system((string("ln -s ") + fname + softlink).c_str()) == 0) {
              cerr << "Created " << softlink << " as a soft link to " << fname 
                   << " for convenience." << endl;
            }
            softlinkCreated = true;
          }
        }
      }
    }//while
  } // should do training?
  if (true) { // do test evaluation
    double llh = 0;
    double trs = 0;
    double right = 0;
    //double correct_heads = 0;
    //double total_heads = 0;
    std::vector<std::vector<std::vector<string>>> refs, hyps;
    auto t_start = std::chrono::high_resolution_clock::now();
    unsigned corpus_size = corpus.nsentencesDev;

    int miss_head = 0;

    for (unsigned sii = 0; sii < corpus_size; ++sii) {
      const std::vector<unsigned>& sentence=corpus.sentencesDev[sii];
      const std::vector<unsigned>& sentencePos=corpus.sentencesPosDev[sii]; 
      const std::vector<string>& sentenceUnkStr=corpus.sentencesStrDev[sii]; 
      const std::vector<unsigned>& actions=corpus.correct_act_sentDev[sii];
      std::vector<unsigned> tsentence=sentence;
      for (auto& w : tsentence)
        if (training_vocab.count(w) == 0) w = kUNK;
      double lp = 0;
      std::vector<unsigned> pred;
      std::vector<std::vector<string>> cand;
      std::vector<Expression> word_rep; // word representations
      Expression act_rep; // final action representation
      //cerr<<"compute action" << endl;
      {
      ComputationGraph cg;
      pred = parser->log_prob_parser(&cg, sentence, tsentence, sentencePos, std::vector<unsigned>(),
                                                        corpus.actions, corpus.intToWords, &right, cand, &word_rep, &act_rep);
      }
      /*cerr << cand.size() << endl;
      for (unsigned i = 0; i < cand.size(); ++i){
        for (unsigned j = 0; j < cand.size(); ++j){
            if (cand[i][j] != REL_NULL)
                cerr << "from " << i << " to " << j << " rel: " << cand[i][j] << endl;
        }
      }*/
      llh -= lp;
      trs += actions.size();
      //map<int, string> rel_ref, rel_hyp;
      //cerr << "compute heads "<<endl;
      std::vector<std::vector<string>> ref = parser->compute_heads(sentence, actions, corpus.actions);
      std::vector<std::vector<string>> hyp = parser->compute_heads(sentence, pred, corpus.actions);
      refs.push_back(ref);
      hyps.push_back(hyp);
      if (parser->process_headless(hyp, cand, word_rep, act_rep, corpus.actions, sentence, sentencePos) > 0) {
            miss_head++;
            cerr << corpus.intToWords[sentence[0]] << corpus.intToWords[sentence[1]]<< endl;
        }
        //cerr<<"write to file" <<endl;
      output_conll(sentence, sentencePos, sentenceUnkStr, corpus.intToWords, corpus.intToPos, hyp);
      //correct_heads += compute_correct(ref, hyp, sentence.size() - 1);
      //total_heads += sentence.size() - 1;
    }
    //cerr << "miss head number: " << miss_head << endl;
    map<string, double> results = ltp::lstmsdparser::evaluate(refs, hyps, corpus.sentencesPosDev, corpus.posToInt["PU"]);
    auto t_end = std::chrono::high_resolution_clock::now();
    cerr << "TEST llh=" << llh << " ppl: " << exp(llh / trs) << " err: " << (trs - right) / trs 
    << " LF: " << results["LF"] << " UF:" << results["UF"]  << " LP:" << results["LP"] << " LR:" << results["LR"] 
    << " UP:" << results["UP"] << " UR:" <<results["UR"]  << "\t[" << corpus_size << " sents in " 
        << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms]" << endl;
  }
  for (unsigned i = 0; i < corpus.actions.size(); ++i) {
    //cerr << corpus.actions[i] << '\t' << parser.p_r->values[i].transpose() << endl;
    //cerr << corpus.actions[i] << '\t' << parser.p_p2a->values.col(i).transpose() << endl;
  }
}

