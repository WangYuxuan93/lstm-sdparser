#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>
#include <ctime>
#include <time.h>

#include <unordered_map>
#include <unordered_set>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "dynet/training.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/nodes.h"
#include "dynet/lstm.h"
#include "dynet/rnn.h"
#include "lstmsdparser/c2.h"
#include "lstmsdparser/layers.h"
#include "lstmsdparser/theirtreelstm.h"

//#include "lstm-parse.h"

//namespace ltp {
//namespace lstmsdparser {

const std::string REL_NULL = "-NULL-";
const std::string REL_EXIST = "-EXIST-";

cpyp::Corpus corpus;
volatile bool requested_stop = false;
unsigned LAYERS = 2;
unsigned INPUT_DIM = 40;
unsigned HIDDEN_DIM = 60;
unsigned ACTION_DIM = 36;
unsigned PRETRAINED_DIM = 50;
unsigned LSTM_INPUT_DIM = 60;
unsigned POS_DIM = 10;
unsigned REL_DIM = 8;
unsigned BILSTM_HIDDEN_DIM = 32; // [bilstm]
bool use_bilstm = false; //[bilstm]
bool use_treelstm = false; // [treelstm]

std::string transition_system = "list";

bool USE_POS = false;

constexpr const char* ROOT_SYMBOL = "ROOT";
unsigned kROOT_SYMBOL = 0;
unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned POS_SIZE = 0;

using namespace dynet::expr;
using namespace dynet;
using namespace std;
namespace po = boost::program_options;

vector<unsigned> possible_actions;
unordered_map<unsigned, vector<float>> pretrained;

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
        ("use_bilstm,B", "use bilstm for buffer")
        ("use_treelstm,R", "use treelstm for subtree in stack")
        ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
        ("action_dim", po::value<unsigned>()->default_value(16), "action embedding size")
        ("input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
        ("hidden_dim", po::value<unsigned>()->default_value(64), "hidden dimension")
        ("bilstm_hidden_dim", po::value<unsigned>()->default_value(32), "bilstm hidden dimension")
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

bool has_path_to(int w1, int w2, const vector<vector<string>>& graph){
    //cerr << endl << w1 << " has path to " << w2 << endl;
    if (graph[w1][w2] != REL_NULL)
        return true;
    for (int i = 0; i < (int)graph.size(); ++i){
        if (graph[w1][i] != REL_NULL)
            if (has_path_to(i, w2, graph))
                return true;
    }
    return false;
}

bool has_path_to(int w1, int w2, const vector<vector<bool>>& graph){
    //cerr << endl << w1 << " has path to " << w2 << endl;
    if (graph[w1][w2])
        return true;
    for (int i = 0; i < (int)graph.size(); ++i){
        if (graph[w1][i])
            if (has_path_to(i, w2, graph))
                return true;
    }
    return false;
}

struct ParserBuilder {

  LSTMBuilder stack_lstm; // (layers, input, hidden, trainer)
  LSTMBuilder buffer_lstm;
  LSTMBuilder pass_lstm; // lstm for pass buffer
  LSTMBuilder action_lstm;
  BidirectionalLSTMLayer buffer_bilstm; //[bilstm] bilstm for buffer
  TheirTreeLSTMBuilder tree_lstm; // [treelstm] for subtree
  LookupParameter p_w; // word embeddings
  LookupParameter p_t; // pretrained word embeddings (not updated)
  LookupParameter p_a; // input action embeddings
  LookupParameter p_r; // relation embeddings
  LookupParameter p_p; // pos tag embeddings
  Parameter p_pbias; // parser state bias
  Parameter p_A; // action lstm to parser state
  Parameter p_B; // buffer lstm to parser state
  Parameter p_fwB; // [bilstm] buffer forward lstm to parser state
  Parameter p_bwB; // [bilstm] buffer backward lstm to parser state
  Parameter p_P; // pass lstm to parser state
  Parameter p_S; // stack lstm to parser state
  Parameter p_H; // head matrix for composition function
  Parameter p_D; // dependency matrix for composition function
  Parameter p_R; // relation matrix for composition function
  Parameter p_w2l; // word to LSTM input
  Parameter p_p2l; // POS to LSTM input
  Parameter p_t2l; // pretrained word embeddings to LSTM input
  Parameter p_ib; // LSTM input bias
  Parameter p_cbias; // composition function bias
  Parameter p_p2a;   // parser state to action
  Parameter p_action_start;  // action bias
  Parameter p_abias;  // action bias
  Parameter p_buffer_guard;  // end of buffer
  Parameter p_stack_guard;  // end of stack
  Parameter p_pass_guard;  // end of pass buffer

  bool use_pretrained;

  explicit ParserBuilder(Model& model, const unordered_map<unsigned, vector<float>>& pretrained) :
      stack_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model),
      buffer_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model),
      pass_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model),
      action_lstm(LAYERS, ACTION_DIM, HIDDEN_DIM, model),
      p_w(model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
      p_a(model.add_lookup_parameters(ACTION_SIZE, {ACTION_DIM})),
      p_r(model.add_lookup_parameters(ACTION_SIZE, {REL_DIM})),
      p_pbias(model.add_parameters({HIDDEN_DIM})),
      p_A(model.add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_B(model.add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_P(model.add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_S(model.add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_H(model.add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM})),
      p_D(model.add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM})),
      p_R(model.add_parameters({LSTM_INPUT_DIM, REL_DIM})),
      p_w2l(model.add_parameters({LSTM_INPUT_DIM, INPUT_DIM})),
      p_ib(model.add_parameters({LSTM_INPUT_DIM})),
      p_cbias(model.add_parameters({LSTM_INPUT_DIM})),
      p_p2a(model.add_parameters({ACTION_SIZE, HIDDEN_DIM})),
      p_action_start(model.add_parameters({ACTION_DIM})),
      p_abias(model.add_parameters({ACTION_SIZE})),
      p_buffer_guard(model.add_parameters({LSTM_INPUT_DIM})),
      p_stack_guard(model.add_parameters({LSTM_INPUT_DIM})),
      p_pass_guard(model.add_parameters({LSTM_INPUT_DIM})) {
    if (use_bilstm) {
      buffer_bilstm = BidirectionalLSTMLayer(model, LAYERS, LSTM_INPUT_DIM, BILSTM_HIDDEN_DIM);
      p_fwB = model.add_parameters({HIDDEN_DIM, BILSTM_HIDDEN_DIM});
      p_bwB = model.add_parameters({HIDDEN_DIM, BILSTM_HIDDEN_DIM});
      cerr << "Created Buffer BiLSTM" << endl;
    }
    if (use_treelstm) {
      tree_lstm = TheirTreeLSTMBuilder(1, LSTM_INPUT_DIM, LSTM_INPUT_DIM, model);
      cerr << "Created TreeLSTM" << endl;
    }
    if (USE_POS) {
      p_p = model.add_lookup_parameters(POS_SIZE, {POS_DIM});
      p_p2l = model.add_parameters({LSTM_INPUT_DIM, POS_DIM});
    }
    if (pretrained.size() > 0) {
      use_pretrained = true;
      p_t = model.add_lookup_parameters(VOCAB_SIZE, {PRETRAINED_DIM});
      for (auto it : pretrained)
        p_t.initialize(it.first, it.second);
      p_t2l = model.add_parameters({LSTM_INPUT_DIM, PRETRAINED_DIM});
    } else {
      use_pretrained = false;
      //p_t = nullptr;
      //p_t2l = nullptr;
    }
  }

/*
static bool IsActionForbidden(const string& a, unsigned bsize, unsigned ssize, const vector<int>& stacki) {
  if (a[1]=='W' && ssize<3) return true;
  if (a[1]=='W') {
        int top=stacki[stacki.size()-1];
        int sec=stacki[stacki.size()-2];
        if (sec>top) return true;
  }

  bool is_shift = (a[0] == 'S' && a[1]=='H');
  bool is_reduce = !is_shift;
  if (is_shift && bsize == 1) return true;
  if (is_reduce && ssize < 3) return true;
  if (bsize == 2 && // ROOT is the only thing remaining on buffer
      ssize > 2 && // there is more than a single element on the stack
      is_shift) return true;
  // only attach left to ROOT
  if (bsize == 1 && ssize == 3 && a[0] == 'R') return true;
  return false;
}*/

static bool IsActionForbidden(const string& a, unsigned bsize, unsigned ssize, unsigned root, 
                              const vector<vector<bool>> dir_graph, const vector<int>& stacki, 
                              const vector<int>& bufferi) {
    if (transition_system == "list"){
        //cerr << a << endl;
        int s0 = stacki.back();
        int b0 = bufferi.back();
        int root_num = 0;
        int s0_head_num = 0;
        for (int i = 0; i < (int)dir_graph[root].size(); ++i)
            if (dir_graph[root][i])
                root_num ++;
        if (s0 >= 0)
            for (int i = 0; i < (int)dir_graph[root].size(); ++i)
                if (dir_graph[i][s0])
                    s0_head_num ++;
        if (a[0] == 'L'){
            string rel = a.substr(3, a.size() - 4);
            if (bsize < 2 || ssize < 2) return true;
            if (has_path_to(s0, b0, dir_graph)) return true;
            //if (b0 == root && rel != "Root") return true;
            //if (b0 == root && rel == "Root" && root_num >= 1) return true;
            if (b0 == (int)root && !(rel == "Root" && root_num == 0 && s0_head_num == 0)) return true;
            if (b0 != (int)root && rel == "Root") return true;
        }
        if (a[0] == 'R'){
            if (bsize < 2 || ssize < 2) return true;
            if (has_path_to(b0, s0, dir_graph)) return true;
            if (b0 == (int)root) return true;
        }
        if (a[0] == 'N'){
            if (a[1] == 'S' && bsize < 2) return true;
            //if (a[1] == 'S' && bsize == 2 && ssize > 2) return true;
            if (a[1] == 'R' && !(ssize > 1 && s0_head_num > 0)) return true;
            if (a[1] == 'P' && !(ssize > 1 && bsize > 1))  return true;
        }
        return false;
  }
    else if (transition_system == "spl"){
        int s0 = stacki.back();
        int b0 = bufferi.back();
        int root_num = 0;
        int s0_head_num = 0;
        for (int i = 0; i < (int)dir_graph[root].size(); ++i)
            if (dir_graph[root][i])
                root_num ++;
        if (s0 >= 0)
            for (int i = 0; i < (int)dir_graph[root].size(); ++i)
                if (dir_graph[i][s0])
                    s0_head_num ++;
        if (a[0] == 'L'){
            string rel = a.substr(3, a.size() - 4);
            if (bsize < 2 || ssize < 2) return true;
            if (has_path_to(s0, b0, dir_graph)) return true;
            //if (b0 == root && rel != "Root") return true;
            //if (b0 == root && rel == "Root" && root_num >= 1) return true;
            if (b0 == (int)root && !(rel == "Root" && root_num == 0 && s0_head_num == 0)) return true;
            if (b0 != (int)root && rel == "Root") return true;
        }
        if (a[0] == 'R'){
            if (bsize < 2 || ssize < 2) return true;
            if (has_path_to(b0, s0, dir_graph)) return true;
            if (b0 == (int)root) return true;
        }
        if (a[0] == 'N'){
            if (a[1] == 'S' && bsize < 2) return true;
            //if (a[1] == 'S' && bsize == 2 && ssize > 2) return true;
            if (a[1] == 'P' && !(ssize > 1 && bsize > 1))  return true;
        }
        return false;
  }
    return false;
}

static vector<vector<string>> compute_heads(const vector<unsigned>& sent, const vector<unsigned>& actions, 
                                                                                    const vector<string>& setOfActions) {
  //map<int,int> heads;
  //map<int,string> r;
  //map<int,string>& rels = (pr ? *pr : r);
    unsigned sent_len = sent.size();
    vector<vector<string>> graph;
     
    for (unsigned i = 0; i < sent_len; i++) {
        vector<string> r;
        for (unsigned j = 0; j < sent_len; j++) r.push_back(REL_NULL);
        graph.push_back(r);
    }
    vector<int> bufferi(sent_len + 1, 0), stacki(1, -999), passi(1, -999);
    for (unsigned i = 0; i < sent_len; ++i)
        bufferi[sent_len - i] = i;
    bufferi[0] = -999;
    for (auto action: actions) { // loop over transitions for sentence
        const string& actionString=setOfActions[action];
        const char ac = actionString[0];
        const char ac2 = actionString[1];

      /*  cerr <<endl<<"[";
      for (int i = (int)stacki.size() - 1; i > -1 ; --i)
        cerr << corpus.intToWords[sent[stacki[i]]] <<"-"<<stacki[i]<<", ";
      cerr <<"][";
      for (int i = (int)passi.size() - 1; i > -1 ; --i)
        cerr << corpus.intToWords[sent[passi[i]]]<<"-"<<passi[i]<<", ";
      cerr <<"][";
      for (int i = (int)bufferi.size() - 1; i > -1 ; --i)
        cerr << corpus.intToWords[sent[bufferi[i]]]<<"-"<<bufferi[i]<<", ";
      cerr <<"]"<<endl;

      cerr << "action:" << actionString << endl;*/

        if (transition_system == "list"){
            if (ac =='N' && ac2=='S') {  // NO-SHIFT
                assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
                int passi_size = (int)passi.size();
                for (int i = 1; i < passi_size; i++){  //do not move pass_guard
                    stacki.push_back(passi.back());
                    passi.pop_back();
                }
                stacki.push_back(bufferi.back());
                bufferi.pop_back();
            } else if (ac=='N' && ac2=='R'){
                assert(stacki.size() > 1);
                stacki.pop_back();
            } else if (ac=='N' && ac2=='P'){
                assert(stacki.size() > 1);
                passi.push_back(stacki.back());
                stacki.pop_back();
            } else if (ac=='L'){ // LEFT-REDUCE or LEFT-PASS
                assert(stacki.size() > 1 && bufferi.size() > 1);
                unsigned depi, headi;
                depi = stacki.back();
                stacki.pop_back();
                headi = bufferi.back();
                graph[headi][depi] = actionString.substr(3, actionString.size() - 4);
                if (ac2 == 'P'){ // LEFT-PASS
                    //TODO pass_lstm
                    passi.push_back(depi);
                }
            } else if (ac=='R'){ // RIGHT-SHIFT or RIGHT-PASSA
                assert(stacki.size() > 1 && bufferi.size() > 1);
                unsigned depi, headi;
                depi = bufferi.back();
                bufferi.pop_back();
                headi = stacki.back();
                stacki.pop_back();
                graph[headi][depi] = actionString.substr(3, actionString.size() - 4);
                if (ac2 == 'S'){ //RIGHT-SHIFT
                    stacki.push_back(headi);
                    int passi_size = (int)passi.size();
                    for (int i = 1; i < passi_size; i++){  //do not move pass_guard
                        stacki.push_back(passi.back());
                        passi.pop_back();
                    }
                    stacki.push_back(depi);
                }
                else if (ac2 == 'P'){
                    //TODO pass_lstm.add_input(nlcomposed);
                    passi.push_back(headi);
                    bufferi.push_back(depi);
                }
            }
        }
        else if (transition_system == "spl"){
            //TODO
            if (ac =='N' && ac2=='S') {  // NO-SHIFT
                assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
                int passi_size = (int)passi.size();
                for (int i = 1; i < passi_size; i++){  //do not move pass_guard
                    stacki.push_back(passi.back());
                    passi.pop_back();
                }
                stacki.push_back(bufferi.back());
                bufferi.pop_back();
            } else if (ac=='N' && ac2=='P'){
                assert(stacki.size() > 1);
                passi.push_back(stacki.back());
                stacki.pop_back();
            } else if (ac=='L'){ // LEFT-ARC or LEFT-POP
                assert(stacki.size() > 1 && bufferi.size() > 1);
                unsigned depi, headi;
                depi = stacki.back();
                stacki.pop_back();
                headi = bufferi.back();
                graph[headi][depi] = actionString.substr(3, actionString.size() - 4);
                if (ac2 == 'A'){ // LEFT-ARC
                    //TODO pass_lstm
                    passi.push_back(depi);
                }
            } else if (ac=='R'){ // RIGHT-ARC
                assert(stacki.size() > 1 && bufferi.size() > 1);
                unsigned depi, headi;
                depi = bufferi.back();
                bufferi.pop_back();
                headi = stacki.back();
                stacki.pop_back();
                graph[headi][depi] = actionString.substr(3, actionString.size() - 4);
                passi.push_back(headi);
                bufferi.push_back(depi);
            }
        }
  }
  assert(bufferi.size() == 1);
  //assert(stacki.size() == 2);
  return graph;
}

vector<unsigned> get_children(unsigned id, const vector<vector<bool>> graph){
  vector<unsigned> children;
  for (int i = 0; i < unsigned(graph[0].size()); i++){
    if (graph[id][i])
      children.push_back(i);
  }
  return children;
}

// *** if correct_actions is empty, this runs greedy decoding ***
// returns parse actions for input sentence (in training just returns the reference)
// OOV handling: raw_sent will have the actual words
//               sent will have words replaced by appropriate UNK tokens
// this lets us use pretrained embeddings, when available, for words that were OOV in the
// parser training data
vector<unsigned> log_prob_parser(ComputationGraph* hg,
                     const vector<unsigned>& raw_sent,  // raw sentence
                     const vector<unsigned>& sent,  // sent with oovs replaced
                     const vector<unsigned>& sentPos,
                     const vector<unsigned>& correct_actions,
                     const vector<string>& setOfActions,
                     const map<unsigned, std::string>& intToWords,
                     double *right, 
                     vector<vector<string>>& cand,
                     vector<Expression>* word_rep = NULL,
                     Expression * act_rep = NULL) {
    vector<unsigned> results;
    const bool build_training_graph = correct_actions.size() > 0;
    //init word representation
    if (word_rep){
        for (unsigned i = 0; i < sent.size(); ++i){
            Expression wd;
            (*word_rep).push_back(wd);
        }
    }
    //init candidate
    vector<string> cv;
    for (unsigned i = 0; i < sent.size(); ++i)
        cv.push_back(REL_NULL);
    for (unsigned i = 0; i < sent.size(); ++i)
        cand.push_back(cv);

    stack_lstm.new_graph(*hg);
    pass_lstm.new_graph(*hg);
    action_lstm.new_graph(*hg);
    stack_lstm.start_new_sequence();
    pass_lstm.start_new_sequence();
    action_lstm.start_new_sequence();

    Expression fwB;
    Expression bwB;
    if (use_bilstm){
      buffer_bilstm.new_graph(hg); // [bilstm] start_new_sequence is implemented in add_input
      fwB = parameter(*hg, p_fwB); // [bilstm]
      bwB = parameter(*hg, p_bwB); // [bilstm]
    }else{
      buffer_lstm.new_graph(*hg);
      buffer_lstm.start_new_sequence();
    }
    if (use_treelstm){
      tree_lstm.new_graph(*hg); // [treelstm]
      tree_lstm.start_new_sequence(); // [treelstm]
      tree_lstm.initialize_structure(sent.size()); // [treelstm]
    }
    // variables in the computation graph representing the parameters
    Expression pbias = parameter(*hg, p_pbias);
    Expression H = parameter(*hg, p_H);
    Expression D = parameter(*hg, p_D);
    Expression R = parameter(*hg, p_R);
    Expression cbias = parameter(*hg, p_cbias);
    Expression S = parameter(*hg, p_S);
    Expression B = parameter(*hg, p_B);
    Expression P = parameter(*hg, p_P);
    Expression A = parameter(*hg, p_A);
    Expression ib = parameter(*hg, p_ib);
    Expression w2l = parameter(*hg, p_w2l);
    Expression p2l;

    //cerr << "w2l: " << w2l.dim().ndims() << w2l.dim().d[0] << w2l.dim().d[1] << endl;

    if (USE_POS)
      p2l = parameter(*hg, p_p2l);
    Expression t2l;
    if (use_pretrained)
      t2l = parameter(*hg, p_t2l);
    Expression p2a = parameter(*hg, p_p2a);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);

    action_lstm.add_input(action_start);
    vector<Expression> buffer(sent.size() + 1);  // variables representing word embeddings (possibly including POS info)
    vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence
    // precompute buffer representation from left to right
    vector<Expression> word_emb(sent.size()); // [treelstm] store original word representation emb[i] for sent[i]

    for (unsigned i = 0; i < sent.size(); ++i) {
      assert(sent[i] < VOCAB_SIZE);
      Expression w =lookup(*hg, p_w, sent[i]);

      vector<Expression> args = {ib, w2l, w}; // learn embeddings
      if (USE_POS) { // learn POS tag?
        Expression p = lookup(*hg, p_p, sentPos[i]);
        args.push_back(p2l);
        args.push_back(p);
      }
      if (use_pretrained && pretrained.count(raw_sent[i])) {  // include fixed pretrained vectors?
        Expression t = const_lookup(*hg, p_t, raw_sent[i]);
        args.push_back(t2l);
        args.push_back(t);
      }
      //buffer[] = ib + w2l * w + p2l * p + t2l * t
      buffer[sent.size() - i] = rectify(affine_transform(args));
      bufferi[sent.size() - i] = i;
      if (use_treelstm){
        word_emb[i] = buffer[sent.size() - i];
      }
    }
    if (use_treelstm){
      vector<unsigned> h;
      for (int i = 0; i < sent.size(); i++)
        tree_lstm.add_input(i, h, word_emb[i]);
    }

    // dummy symbol to represent the empty buffer
    buffer[0] = parameter(*hg, p_buffer_guard);
    bufferi[0] = -999;
    std::vector<BidirectionalLSTMLayer::Output> bilstm_outputs;
    if (use_bilstm){
      buffer_bilstm.add_inputs(hg, buffer); 
      buffer_bilstm.get_outputs(hg, bilstm_outputs); // [bilstm] output of bilstm for buffer, first is fw, second is bw
    }else{
      for (auto& b : buffer)
        buffer_lstm.add_input(b);
    }
    vector<Expression> pass; //variables reperesenting embedding in pass buffer
    vector<int> passi; //position of words in pass buffer
    pass.push_back(parameter(*hg, p_pass_guard));
    passi.push_back(-999); // not used for anything
    pass_lstm.add_input(pass.back());

    vector<Expression> stack;  // variables representing subtree embeddings
    vector<int> stacki; // position of words in the sentence of head of subtree
    stack.push_back(parameter(*hg, p_stack_guard));
    stacki.push_back(-999); // not used for anything
    // drive dummy symbol on stack through LSTM
    stack_lstm.add_input(stack.back());
    vector<Expression> log_probs;
    string rootword;
    unsigned action_count = 0;  // incremented at each prediction

    //init graph connecting vector
    //vector<bool> dir_graph[sent.size()]; // store the connection between words in sent
    vector<vector<bool>> dir_graph;
    vector<bool> v;
    for (int i = 0; i < (int)sent.size(); i++){
        v.push_back(false);
    }
    for (int i = 0; i < (int)sent.size(); i++){
        dir_graph.push_back(v);
    }
    //remove stack.size() > 2 ||
    while( bufferi.size() > 1) {
      // get list of possible actions for the current parser state
      vector<unsigned> current_valid_actions;
      /*if (!build_training_graph){
      cerr << sent.size() << endl;
      cerr <<endl<<"[";
      for (int i = (int)stacki.size() - 1; i > -1 ; --i)
        cerr << corpus.intToWords[sent[stacki[i]]] <<"-"<<stacki[i]<<", ";
      cerr <<"][";
      for (int i = (int)passi.size() - 1; i > -1 ; --i)
        cerr << corpus.intToWords[sent[passi[i]]]<<"-"<<passi[i]<<", ";
      cerr <<"][";
      for (int i = (int)bufferi.size() - 1; i > -1 ; --i)
        cerr << corpus.intToWords[sent[bufferi[i]]]<<"-"<<bufferi[i]<<", ";
      //cerr <<"]"<<endl;
      //}*/
      for (auto a: possible_actions) {
        //cerr << " " << setOfActions[a]<< " ";
        if (IsActionForbidden(setOfActions[a], bufferi.size(), stacki.size(), sent.size() - 1, dir_graph, stacki, bufferi))
          continue;
        //cerr << " <" << setOfActions[a] << "> ";
        current_valid_actions.push_back(a);
      }
      Expression p_t;
      if (use_bilstm){
        // [bilstm] p_t = pbias + S * slstm + P * plstm + fwB * blstm_fw + bwB * blstm_bw + A * almst
        p_t = affine_transform({pbias, S, stack_lstm.back(), P, pass_lstm.back(), 
          fwB, bilstm_outputs[sent.size() - bufferi.back()].first, bwB, bilstm_outputs[sent.size() - bufferi.back()].second,
          A, action_lstm.back()});
        //cerr << " bilstm: " << sent.size() - bufferi.back() << endl;
      }else{
        // p_t = pbias + S * slstm + P * plstm + B * blstm + A * almst
        p_t = affine_transform({pbias, S, stack_lstm.back(), P, pass_lstm.back(), B, buffer_lstm.back(), A, action_lstm.back()});
      }
      
      Expression nlp_t = rectify(p_t);
      // r_t = abias + p2a * nlp
      Expression r_t = affine_transform({abias, p2a, nlp_t});

      // adist = log_softmax(r_t, current_valid_actions)
      Expression adiste = log_softmax(r_t, current_valid_actions);
      vector<float> adist = as_vector(hg->incremental_forward(adiste));
      double best_score = adist[current_valid_actions[0]];
      unsigned best_a = current_valid_actions[0];
      for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
        if (adist[current_valid_actions[i]] > best_score) {
          best_score = adist[current_valid_actions[i]];
          best_a = current_valid_actions[i];
        }
      }
      unsigned action = best_a;
      if (build_training_graph) {  // if we have reference actions (for training) use the reference action
        action = correct_actions[action_count];
        if (best_a == action) { (*right)++; }
      }
      if (setOfActions[action] == "NS"){
        double second_score = - DBL_MAX;
        string second_a = setOfActions[current_valid_actions[0]];
        for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
            const string& actionstr=setOfActions[current_valid_actions[i]];
            const char  ac0 = actionstr[0];
            //cerr << actionstr << "-" << adist[current_valid_actions[i]] << endl;
            if (adist[current_valid_actions[i]] > second_score &&
                (ac0 == 'R' || ac0 == 'L')) {
                second_score = adist[current_valid_actions[i]];
                second_a = actionstr;
            }
        }
        //cerr << second_a << endl;
        if (second_a[0] == 'L' || second_a[0] == 'R'){
            int headi = (second_a[0] == 'L' ? bufferi.back() : stacki.back());
            int depi = (second_a[0] == 'L' ? stacki.back() : bufferi.back());
            cand[headi][depi] = second_a.substr(3, second_a.size() - 4);
        }
      }
      /*if (!build_training_graph)
        cerr <<endl<< "gold action: " << setOfActions[action] <<endl;*/
      ++action_count;
      log_probs.push_back(pick(adiste, action));
      results.push_back(action);

      // add current action to action LSTM
      Expression actione = lookup(*hg, p_a, action);
      action_lstm.add_input(actione);

      // get relation embedding from action (TODO: convert to relation from action?)
      Expression relation = lookup(*hg, p_r, action);

      // do action
      const string& actionString=setOfActions[action];
      const char ac = actionString[0];
      const char ac2 = actionString[1];
      //cerr << ac << ac2 << endl;

        if (transition_system == "list"){
            if (ac =='N' && ac2=='S') {  // NO-SHIFT
                assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
                int pass_size = (int)pass.size();
                for (int i = 1; i < pass_size; i++){  //do not move pass_guard
                    stack.push_back(pass.back());
                    stack_lstm.add_input(pass.back());
                    pass.pop_back();
                    pass_lstm.rewind_one_step();
                    stacki.push_back(passi.back());
                    passi.pop_back();
                }
                stack.push_back(buffer.back());
                stack_lstm.add_input(buffer.back());
                buffer.pop_back();
                if (!use_bilstm)
                  buffer_lstm.rewind_one_step();
                stacki.push_back(bufferi.back());
                bufferi.pop_back();
            } else if (ac=='N' && ac2=='R'){
                assert(stacki.size() > 1);
                if (word_rep)
                    (*word_rep)[stacki.back()] = stack.back();
                stack.pop_back();
                stacki.pop_back();
                stack_lstm.rewind_one_step();
            } else if (ac=='N' && ac2=='P'){
                assert(stacki.size() > 1);
                pass.push_back(stack.back());
                pass_lstm.add_input(stack.back());
                passi.push_back(stacki.back());
                stack.pop_back();
                stacki.pop_back();
                stack_lstm.rewind_one_step();
            } else if (ac=='L'){ // LEFT-REDUCE or LEFT-PASS
                assert(stacki.size() > 1 && bufferi.size() > 1);
                Expression dep, head;
                unsigned depi, headi;
                dep = stack.back();
                depi = stacki.back();
                stack.pop_back();
                stacki.pop_back();
                head = buffer.back();
                headi = bufferi.back();
                buffer.pop_back();
                bufferi.pop_back();
                dir_graph[headi][depi] = true; // add this arc to graph
                //dir_graph[headi][depi] = REL_EXIST;
                if (headi == sent.size() - 1) rootword = intToWords.find(sent[depi])->second;
                Expression nlcomposed;
                if (use_treelstm){
                  vector<unsigned> c = get_children(headi, dir_graph);
                  /*cerr << "children: ";
                  for(int i = 0; i < c.size(); i++)
                    cerr << c[i] << " ";
                  cerr << endl;*/
                  nlcomposed = tree_lstm.add_input(headi, get_children(headi, dir_graph), word_emb[headi]);
                } else{
                  Expression composed = affine_transform({cbias, H, head, D, dep, R, relation});
                  nlcomposed = tanh(composed);
                }
                stack_lstm.rewind_one_step();
                if (!use_bilstm){
                  buffer_lstm.rewind_one_step();
                  buffer_lstm.add_input(nlcomposed);
                }
                buffer.push_back(nlcomposed);
                bufferi.push_back(headi);
                if (ac2 == 'R'){
                    if (word_rep)
                        (*word_rep)[depi] = dep;
                }
                if (ac2 == 'P'){ // LEFT-PASS
                    pass_lstm.add_input(dep);
                    pass.push_back(dep);
                    passi.push_back(depi);
                }
            } else if (ac=='R'){ // RIGHT-SHIFT or RIGHT-PASSA
                assert(stacki.size() > 1 && bufferi.size() > 1);
                Expression dep, head;
                unsigned depi, headi;
                dep = buffer.back();
                depi = bufferi.back();
                buffer.pop_back();
                bufferi.pop_back();
                head = stack.back();
                headi = stacki.back();
                stack.pop_back();
                stacki.pop_back();
                dir_graph[headi][depi] = true; // add this arc to graph
                //dir_graph[headi][depi] = REL_EXIST;
                if (headi == sent.size() - 1) rootword = intToWords.find(sent[depi])->second;
                Expression nlcomposed;
                if (use_treelstm){
                  vector<unsigned> c = get_children(headi, dir_graph);
                  /*cerr << "children: ";
                  for(int i = 0; i < c.size(); i++)
                    cerr << c[i] << " ";
                  cerr << endl;*/
                  nlcomposed = tree_lstm.add_input(headi, get_children(headi, dir_graph),word_emb[headi]);
                } else{
                  Expression composed = affine_transform({cbias, H, head, D, dep, R, relation});
                  nlcomposed = tanh(composed);
                }
                stack_lstm.rewind_one_step();
                if (!use_bilstm)
                    buffer_lstm.rewind_one_step();
                if (ac2 == 'S'){ //RIGHT-SHIFT
                    stack_lstm.add_input(nlcomposed);
                    stack.push_back(nlcomposed);
                    stacki.push_back(headi);
                    int pass_size = (int)pass.size();
                    for (int i = 1; i < pass_size; i++){  //do not move pass_guard
                        stack.push_back(pass.back());
                        stack_lstm.add_input(pass.back());
                        pass.pop_back();
                        pass_lstm.rewind_one_step();
                        stacki.push_back(passi.back());
                        passi.pop_back();
                    }
                    stack_lstm.add_input(dep);
                    stack.push_back(dep);
                    stacki.push_back(depi);
                }
                else if (ac2 == 'P'){
                    pass_lstm.add_input(nlcomposed);
                    pass.push_back(nlcomposed);
                    passi.push_back(headi);
                    if (!use_bilstm)
                      buffer_lstm.add_input(dep);
                    buffer.push_back(dep);
                    bufferi.push_back(depi);
                }
            }
        }
        else if (transition_system == "spl"){
            //TODO
            if (ac =='N' && ac2=='S') {  // NO-SHIFT
                assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
                int pass_size = (int)pass.size();
                for (int i = 1; i < pass_size; i++){  //do not move pass_guard
                    stack.push_back(pass.back());
                    stack_lstm.add_input(pass.back());
                    pass.pop_back();
                    pass_lstm.rewind_one_step();
                    stacki.push_back(passi.back());
                    passi.pop_back();
                }
                stack.push_back(buffer.back());
                stack_lstm.add_input(buffer.back());
                buffer.pop_back();
                if (!use_bilstm)
                  buffer_lstm.rewind_one_step();
                stacki.push_back(bufferi.back());
                bufferi.pop_back();
            } else if (ac=='N' && ac2=='P'){
                assert(stacki.size() > 1);
                pass.push_back(stack.back());
                pass_lstm.add_input(stack.back());
                passi.push_back(stacki.back());
                stack.pop_back();
                stacki.pop_back();
                stack_lstm.rewind_one_step();
            } else if (ac=='L'){ // LEFT-ARC or LEFT-POP
                assert(stacki.size() > 1 && bufferi.size() > 1);
                Expression dep, head;
                unsigned depi, headi;
                dep = stack.back();
                depi = stacki.back();
                stack.pop_back();
                stacki.pop_back();
                head = buffer.back();
                headi = bufferi.back();
                buffer.pop_back();
                bufferi.pop_back();
                dir_graph[headi][depi] = true; // add this arc to graph
                //dir_graph[headi][depi] = REL_EXIST;
                if (headi == sent.size() - 1) rootword = intToWords.find(sent[depi])->second;
                Expression composed = affine_transform({cbias, H, head, D, dep, R, relation});
                Expression nlcomposed = tanh(composed);
                stack_lstm.rewind_one_step();
                if (!use_bilstm){
                    buffer_lstm.rewind_one_step();
                    buffer_lstm.add_input(nlcomposed);
                }
                
                buffer.push_back(nlcomposed);
                bufferi.push_back(headi);
                if (ac2 == 'P'){
                    if (word_rep)
                        (*word_rep)[depi] = dep;
                }
                if (ac2 == 'A'){ // LEFT-ARC
                    pass_lstm.add_input(dep);
                    pass.push_back(dep);
                    passi.push_back(depi);
                }
            } else if (ac=='R'){ // RIGHT-ARC
                assert(stacki.size() > 1 && bufferi.size() > 1);
                Expression dep, head;
                unsigned depi, headi;
                dep = buffer.back();
                depi = bufferi.back();
                buffer.pop_back();
                bufferi.pop_back();
                head = stack.back();
                headi = stacki.back();
                stack.pop_back();
                stacki.pop_back();
                dir_graph[headi][depi] = true; // add this arc to graph
                //dir_graph[headi][depi] = REL_EXIST;
                if (headi == sent.size() - 1) rootword = intToWords.find(sent[depi])->second;
                Expression composed = affine_transform({cbias, H, head, D, dep, R, relation});
                Expression nlcomposed = tanh(composed);

                stack_lstm.rewind_one_step();
                if (!use_bilstm){
                  buffer_lstm.rewind_one_step();
                  buffer_lstm.add_input(dep);
                }
                pass_lstm.add_input(nlcomposed);
                pass.push_back(nlcomposed);
                passi.push_back(headi);
                buffer.push_back(dep);
                bufferi.push_back(depi);
            }
        }
    }
    //assert(stack.size() == 2); // guard symbol, root
    //assert(stacki.size() == 2);
    assert(buffer.size() == 1); // guard symbol
    assert(bufferi.size() == 1);
    Expression tot_neglogprob = -sum(log_probs);
    assert(tot_neglogprob.pg != nullptr);
    //collect word representation 
    if (word_rep)
        for (unsigned i = 1; i < stack.size(); ++i) // do not collect stack_guard
            (*word_rep)[stacki[i]] = stack[i];
    //collect action representation
    if (act_rep)
        *act_rep = action_lstm.back();
    return results;
  }

  void process_headless_search_all(const vector<unsigned>& sent, const vector<unsigned>& sentPos, 
                                                        const vector<string>& setOfActions, vector<Expression>& word_rep, 
                                                        Expression& act_rep, int n, int sent_len, int dir, map<int, double>* scores, 
                                                        map<int, string>* rels){
        for (int i = n + dir; i >= 0 && i < sent_len; i += dir){
            int s0 = (dir > 0 ? n : i);
            int b0 = (dir > 0 ? i : n);
            double score;
            string rel;
            ComputationGraph cg;
            get_best_label(sent, sentPos, &cg, setOfActions, s0, b0, word_rep , act_rep, sent_len, dir, &score, &rel);
            (*scores)[i] = score;
            (*rels)[i] = rel;
            //cerr << "search all n: " << n << " i: " << i << "rel: " << rel << endl;
        }
  } 

  void get_best_label(const vector<unsigned>& sent, const vector<unsigned>& sentPos, 
                                    ComputationGraph* hg, const vector<string>& setOfActions, 
                                    int s0, int b0, vector<Expression>& word_rep, Expression& act_rep, int sent_size, 
                                    int dir, double *score, string *rel) {
    char prefix = (dir > 0 ? 'L' : 'R');
    //init graph connecting vector
    vector<vector<bool>> dir_graph;
    vector<bool> v;
    for (int i = 0; i < sent_size; i++){
        v.push_back(false);
    }
    for (int i = 0; i < sent_size; i++){
        dir_graph.push_back(v);
    }
    stack_lstm.new_graph(*hg);
    buffer_lstm.new_graph(*hg);
    pass_lstm.new_graph(*hg);
    action_lstm.new_graph(*hg);
    stack_lstm.start_new_sequence();
    buffer_lstm.start_new_sequence();
    pass_lstm.start_new_sequence();
    action_lstm.start_new_sequence();
    // variables in the computation graph representing the parameters
    Expression pbias = parameter(*hg, p_pbias);
    Expression S = parameter(*hg, p_S);
    Expression B = parameter(*hg, p_B);
    Expression P = parameter(*hg, p_P);
    Expression A = parameter(*hg, p_A);
    Expression ib = parameter(*hg, p_ib);
    Expression w2l = parameter(*hg, p_w2l);
    Expression p2l;
    if (USE_POS)
      p2l = parameter(*hg, p_p2l);
    Expression t2l;
    if (use_pretrained)
      t2l = parameter(*hg, p_t2l);
    Expression p2a = parameter(*hg, p_p2a);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);

    action_lstm.add_input(action_start);

    //action_lstm.add_input(act_rep);
    vector<Expression> buffer;  // variables representing word embeddings (possibly including POS info)
    vector<int> bufferi;
    //cerr  << "ppppps 5" << endl;
    Expression w =lookup(*hg, p_w, sent[b0]);

    vector<Expression> args = {ib, w2l, w}; // learn embeddings
    if (USE_POS) { // learn POS tag?
        Expression p = lookup(*hg, p_p, sentPos[b0]);
        args.push_back(p2l);
        args.push_back(p);
    }
    if (use_pretrained && pretrained.count(sent[b0])) {  // include fixed pretrained vectors?
        Expression t = const_lookup(*hg, p_t, sent[b0]);
        args.push_back(t2l);
        args.push_back(t);
    }
    //buffer[] = ib + w2l * w + p2l * p + t2l * t

    buffer.push_back(parameter(*hg, p_buffer_guard));
    buffer.push_back(rectify(affine_transform(args)));
    //cerr  << "ppppps 6" << endl;
    bufferi.push_back(-999);
    bufferi.push_back(b0);
    for (auto& b : buffer)
        buffer_lstm.add_input(b);

    
    vector<Expression> pass; //variables reperesenting embedding in pass buffer
    pass.push_back(parameter(*hg, p_pass_guard));
    pass_lstm.add_input(pass.back());

    //cerr  << "ppppps 3" << endl;

    args = {ib, w2l, w}; // learn embeddings
    if (USE_POS) { // learn POS tag?
        Expression p = lookup(*hg, p_p, sentPos[s0]);
        args.push_back(p2l);
        args.push_back(p);
    }
    if (use_pretrained && pretrained.count(sent[s0])) {  // include fixed pretrained vectors?
        Expression t = const_lookup(*hg, p_t, sent[s0]);
        args.push_back(t2l);
        args.push_back(t);
    }

    vector<Expression> stack;  // variables representing subtree embeddings
    vector<int> stacki;
    stacki.push_back(-999);
    stacki.push_back(s0);
    stack.push_back(parameter(*hg, p_stack_guard));
    stack_lstm.add_input(stack.back());
    stack.push_back(rectify(affine_transform(args)));
    stack_lstm.add_input(stack.back());

    // get list of possible actions for the current parser state
    vector<unsigned> current_valid_actions;
    for (auto a: possible_actions) {
        //cerr << " " << setOfActions[a]<< " ";
        if (IsActionForbidden(setOfActions[a], buffer.size(), stack.size(), sent.size() - 1, dir_graph, stacki, bufferi))
            continue;
        //cerr << " <" << setOfActions[a] << "> ";
        current_valid_actions.push_back(a);
    }
    //cerr  << "ppppps 2" << endl;
    // p_t = pbias + S * slstm + P * plstm + B * blstm + A * almst
    Expression p_t = affine_transform({pbias, S, stack_lstm.back(), P, pass_lstm.back(), B, buffer_lstm.back(), A, action_lstm.back()});
    Expression nlp_t = rectify(p_t);
    //cerr  << "ppppps 1" << endl;
    // r_t = abias + p2a * nlp
    Expression r_t = affine_transform({abias, p2a, nlp_t});

    // adist = log_softmax(r_t, current_valid_actions)
    Expression adiste = log_softmax(r_t, current_valid_actions);
    vector<float> adist = as_vector(hg->incremental_forward(adiste));
    double second_score = - DBL_MAX;
    string second_a = REL_NULL;
    for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
        const string& actionstr=setOfActions[current_valid_actions[i]];
        const char  ac0 = actionstr[0];
        //cerr << actionstr << "-" << adist[current_valid_actions[i]] << endl;
        if (adist[current_valid_actions[i]] > second_score &&
                ac0 == prefix) {
            second_score = adist[current_valid_actions[i]];
            second_a = actionstr;
        }
    }
    // do action
    *score = second_score;
    *rel = second_a;
  }

}; // struct

void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}

map<string, double> evaluate(const vector<vector<vector<string>>>& refs, const vector<vector<vector<string>>>& hyps, 
                                                std::map<int, std::vector<unsigned>>& sentencesPos, const unsigned punc) {
    assert(refs.size() == hyps.size());
    int correct_arcs = 0; // unlabeled
    int correct_arcs_wo_punc = 0;
    int correct_rels = 0; // labeled
    int correct_rels_wo_punc = 0;

    //int correct_labeled_trees = 0;
    int correct_labeled_graphs_wo_punc = 0;
    //int correct_unlabeled_trees = 0;
    int correct_unlabeled_graphs_wo_punc = 0;
    //int correct_root = 0;

    int sum_gold_arcs = 0;
    int sum_gold_arcs_wo_punc = 0;
    int sum_pred_arcs = 0;
    int sum_pred_arcs_wo_punc =0;

    bool correct_labeled_flag_wo_punc = true;
    bool correct_unlabeled_flag_wo_punc = true;

    //int correct_non_local_arcs = 0;
    //int correct_non_local_heads = 0;

    //int sum_non_local_gold_arcs = 0;
    //int sum_non_local_pred_arcs = 0;
    for (int i = 0; i < (int)refs.size(); ++i){
        vector<unsigned> sentPos = sentencesPos[i];
        unsigned sent_len = refs[i].size();
        assert(sentPos.size() == sent_len);
        correct_labeled_flag_wo_punc = true;
        correct_unlabeled_flag_wo_punc = true;
        for (unsigned j = 0; j < sent_len; ++j){
            for (unsigned k = 0; k < sent_len; ++k){
                if (refs[i][j][k] != REL_NULL){
                    sum_gold_arcs ++;
                    //cerr << " id : " << k + 1 << " POS: " << sentPos[k] << endl;
                    if (sentPos[k] != punc)
                        sum_gold_arcs_wo_punc ++;
                    if (hyps[i][j][k] != REL_NULL){
                        correct_arcs ++;
                        if (sentPos[k] != punc)
                            correct_arcs_wo_punc ++;
                        if (hyps[i][j][k] == refs[i][j][k]){
                            correct_rels ++;
                            if (sentPos[k] != punc)
                                correct_rels_wo_punc ++;
                        }
                        else if (sentPos[k] != punc){
                            correct_labeled_flag_wo_punc = false;
                        }
                    }
                    else if (sentPos[k] != punc){
                            correct_labeled_flag_wo_punc = false;
                            correct_unlabeled_flag_wo_punc = false;
                    }
                }
                if (hyps[i][j][k] != REL_NULL){
                    sum_pred_arcs ++;
                    if (sentPos[k] != punc)
                        sum_pred_arcs_wo_punc ++;
                }
            }//k
        }//j
        if (correct_unlabeled_flag_wo_punc){
            correct_unlabeled_graphs_wo_punc ++;
            if (correct_labeled_flag_wo_punc)
                correct_labeled_graphs_wo_punc ++;
        }
    }//i
    //int sum_graphs = (int)refs.size();
    //cerr << "cor: arcs: " << correct_arcs_wo_punc << " rels: " << correct_rels_wo_punc 
            //<< "\nsum: gold arcs: " << sum_gold_arcs_wo_punc << " pred arcs: " << sum_pred_arcs_wo_punc << endl;
    map<string, double> result;
    result["UR"] = correct_arcs_wo_punc * 100.0 / sum_gold_arcs_wo_punc;
    result["UP"] = correct_arcs_wo_punc * 100.0 / sum_pred_arcs_wo_punc;
    result["LR"] = correct_rels_wo_punc * 100.0 / sum_gold_arcs_wo_punc;
    result["LP"] = correct_rels_wo_punc * 100.0 / sum_pred_arcs_wo_punc;

    if (sum_pred_arcs_wo_punc == 0){
        result["LP"] = 0;
        result["UP"] = 0;
    }

    result["UF"] = 2 * result["UR"] * result["UP"] / (result["UR"] + result["UP"]);   
    result["LF"] = 2 * result["LR"] * result["LP"] / (result["LR"] + result["LP"]);
    if (result["LR"] == 0 && result["LP"] == 0)
        result["LF"] = 0;
    if (result["UR"] == 0 && result["UP"] == 0)
        result["UF"] = 0;
    return result;
}


int process_headless(vector<vector<string>>& hyp, vector<vector<string>>& cand, vector<Expression>& word_rep, 
                                    Expression& act_rep, ParserBuilder& parser, const vector<string>& setOfActions, 
                                    const vector<unsigned>& sent, const vector<unsigned>& sentPos){
    //cerr << "process headless" << endl;
    int root = hyp.size() - 1;
    int miss_head_num = 0;
    bool has_head_flag = false;
    for (unsigned i = 0; i < (hyp.size() - 1); ++i){
        has_head_flag = false;
        for (unsigned j = 0; j < hyp.size(); ++j){
            if (hyp[j][i] != REL_NULL)
                has_head_flag = true;
        }
        if (!has_head_flag){
            miss_head_num ++;
            // use candidate relations
            for (unsigned j = 0; j < cand.size(); ++j){
                if (cand[j][i] != REL_NULL && !has_path_to(i, j, hyp)){
                    hyp[j][i] = cand[j][i];
                    miss_head_num --;
                    has_head_flag = true;
                    break;
                }
            }
            // recompute
            if (!has_head_flag){
                //count root
                int root_num = 0;
                for (unsigned q = 0; q < hyp.size() - 1; ++q)
                if (hyp[root][q] != REL_NULL)
                    root_num ++;

                map<int, double> scores;
                map<int, string> rels;
                parser.process_headless_search_all(sent, sentPos, setOfActions, word_rep, act_rep, i, (int)(hyp.size()), 1, &scores, &rels);
                parser.process_headless_search_all(sent, sentPos, setOfActions, word_rep, act_rep, i, (int)(hyp.size()), -1, &scores, &rels);
                if (root_num >0)
                    scores[root] = -DBL_MAX;
                double opt_score = -DBL_MAX;
                int opt_head = -1;
                for (int k = 0; k < (int)i; ++k){
                    if (scores[k] > opt_score
                            && !has_path_to(i, k, hyp)) { // no cycle
                        opt_head = k;
                        opt_score = scores[k];
                    }
                }
                for (int k = (int)i + 1; k < (int)(hyp.size()); ++k){
                    if (scores[k] > opt_score
                            && !has_path_to(i, k, hyp)) { // no cycle
                        opt_head = k;
                        opt_score = scores[k];
                    }
                }
                if (opt_head != -1){
                    //cerr << rels[opt_head] << endl;
                    hyp[opt_head][i] = rels[opt_head].substr(3, rels[opt_head].size() - 4);
                    miss_head_num --;
                }
                /*else{ // show error information
                    cerr << "opt_score " << opt_score <<" opt_head " << opt_head << endl;
                    for (int k = 0; k < (int)i; ++k){
                        cerr << "head: " << k << " score: " << scores[k] << " rel: " << rels[k] << " path: " << has_path_to(i, k, hyp) << endl; 
                    }
                    for (int k = i + 1; k < (int)(hyp.size() - 1); ++k){
                        cerr << "head: " << k << " score: " << scores[k] << " rel: " << rels[k] << " path: " << has_path_to(i, k, hyp) << endl; 
                    }
                }*/
            } // recomput*/
        }
    } // for
    int root_num = 0;
    for (unsigned i = 0; i < hyp.size() - 1; ++i)
        if (hyp[root][i] != REL_NULL)
            root_num ++;
    if (root_num != 1){
        cerr << "root error: " << root_num << endl;
      }
    //cerr << "miss_head_num: " << miss_head_num << endl;
    return miss_head_num;
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
            if (hyp[j][i] != REL_NULL){
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

// } //  namespace lstmsdparser
// } //  namespace ltp


int main(int argc, char** argv) {
  //dynet::Initialize(argc, argv);
  //allocate memory for dynet
  char ** dy_argv = new char * [4];
  int dy_argc = 3;
  dy_argv[0] = "dynet";
  dy_argv[1] = "--dynet-mem";
  dy_argv[2] = "2000";
  //argv[3] = nullptr;
  //auto dyparams = dynet::extract_dynet_params(argc, argv);
  dynet::initialize(dy_argc, dy_argv);
  delete dy_argv;

  cerr << "COMMAND:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 100;

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  USE_POS = conf.count("use_pos_tags");
  use_bilstm = conf.count("use_bilstm");
  use_treelstm = conf.count("use_treelstm");
  if (use_bilstm)
    cerr << "Using bilstm for buffer." << endl;
  if (use_treelstm)
    cerr << "Using treelstm for subtree in stack." << endl;

  transition_system = conf["transition_system"].as<string>();
  cerr << "Transition System: " << transition_system << endl;

  LAYERS = conf["layers"].as<unsigned>();
  INPUT_DIM = conf["input_dim"].as<unsigned>();
  PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
  HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
  ACTION_DIM = conf["action_dim"].as<unsigned>();
  LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
  POS_DIM = conf["pos_dim"].as<unsigned>();
  REL_DIM = conf["rel_dim"].as<unsigned>();

  BILSTM_HIDDEN_DIM = conf["bilstm_hidden_dim"].as<unsigned>(); // [bilstm]
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
  os << "parser_" << (USE_POS ? "pos" : "nopos")
     << '_' << (use_bilstm ? "bilstm" : "nobilstm")
     << '_' << (use_treelstm ? "treelstm" : "notreelstm")
     << '_' << conf["data_type"].as<string>()
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << '_' << ACTION_DIM
     << '_' << LSTM_INPUT_DIM
     << '_' << POS_DIM
     << '_' << REL_DIM
     << "-pid" << getpid() << ".params";
  double best_LF = 0;
  const string fname = os.str();
  cerr << "Writing parameters to file: " << fname << endl;
  bool softlinkCreated = false;
  corpus.load_correct_actions(conf["training_data"].as<string>());	
  const unsigned kUNK = corpus.get_or_add_word(cpyp::Corpus::UNK);
  kROOT_SYMBOL = corpus.get_or_add_word(ROOT_SYMBOL);

  if (conf.count("words")) {
    pretrained[kUNK] = vector<float>(PRETRAINED_DIM, 0);
    cerr << "Loading from " << conf["words"].as<string>() << " with" << PRETRAINED_DIM << " dimensions\n";
    ifstream in(conf["words"].as<string>().c_str());
    string line;
    getline(in, line);
    vector<float> v(PRETRAINED_DIM, 0);
    string word;
    while (getline(in, line)) {
      istringstream lin(line);
      lin >> word;
      for (unsigned i = 0; i < PRETRAINED_DIM; ++i) lin >> v[i];
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
  VOCAB_SIZE = corpus.nwords + 1;
  //ACTION_SIZE = corpus.nactions + 1;
  ACTION_SIZE = corpus.nactions + 30; // leave places for new actions in test set
  POS_SIZE = corpus.npos + 10;  // bad way of dealing with the fact that we may see new POS tags in the test set
  possible_actions.resize(corpus.nactions);
  for (unsigned i = 0; i < corpus.nactions; ++i)
    possible_actions[i] = i;

  Model model;
  ParserBuilder parser(model, pretrained);
  if (conf.count("model")) {
    ifstream in(conf["model"].as<string>().c_str());
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  // OOV words will be replaced by UNK tokens
  corpus.load_correct_actionsDev(conf["dev_data"].as<string>());
  //TRAINING
  if (conf.count("train")) {
    signal(SIGINT, signal_callback_handler);
    SimpleSGDTrainer sgd(model);
    //MomentumSGDTrainer sgd(&model);
    sgd.eta_decay = 0.08;
    //sgd.eta_decay = 0.05;
    vector<unsigned> order(corpus.nsentences);
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
           const vector<unsigned>& sentence=corpus.sentences[order[si]];
           vector<unsigned> tsentence=sentence;
           if (unk_strategy == 1) {
             for (auto& w : tsentence)
               if (singletons.count(w) && dynet::rand01() < unk_prob) w = kUNK;
           }
	   const vector<unsigned>& sentencePos=corpus.sentencesPos[order[si]]; 
	   const vector<unsigned>& actions=corpus.correct_act_sent[order[si]];
           ComputationGraph hg;
           //cerr << "Start word:" << corpus.intToWords[sentence[0]]<<corpus.intToWords[sentence[1]] << endl;
           vector<vector<string>> cand;
           parser.log_prob_parser(&hg,sentence,tsentence,sentencePos,actions,corpus.actions,corpus.intToWords,&right,cand);
           double lp = as_scalar(hg.incremental_forward((VariableIndex)(hg.nodes.size() - 1)));
           if (lp < 0) {
             cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp << endl;
             assert(lp >= 0.0);
           }
           hg.backward((VariableIndex)(hg.nodes.size() - 1));
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
        vector<vector<vector<string>>> refs, hyps;
        for (unsigned sii = 0; sii < dev_size; ++sii) {
           const vector<unsigned>& sentence=corpus.sentencesDev[sii];
	   const vector<unsigned>& sentencePos=corpus.sentencesPosDev[sii]; 
	   const vector<unsigned>& actions=corpus.correct_act_sentDev[sii];
           vector<unsigned> tsentence=sentence;
           for (auto& w : tsentence)
             if (training_vocab.count(w) == 0) w = kUNK;

           ComputationGraph hg;
           vector<vector<string>> cand;
            vector<unsigned> pred = parser.log_prob_parser(&hg,sentence,tsentence,sentencePos,vector<unsigned>(),
                                                                                                corpus.actions,corpus.intToWords,&right,cand);
           double lp = 0;
           llh -= lp;
           trs += actions.size();
           //cerr << "start word:" << sii << corpus.intToWords[sentence[0]] << corpus.intToWords[sentence[1]] << endl;
           vector<vector<string>> ref = parser.compute_heads(sentence, actions, corpus.actions);
           vector<vector<string>> hyp = parser.compute_heads(sentence, pred, corpus.actions);
           //output_conll(sentence, corpus.intToWords, ref, hyp);
           //correct_heads += compute_correct(ref, hyp, sentence.size() - 1);
           //total_heads += sentence.size() - 1;
           refs.push_back(ref);
           hyps.push_back(hyp);
        }
        map<string, double> results = evaluate(refs, hyps, corpus.sentencesPosDev, corpus.posToInt["PU"]);
        auto t_end = std::chrono::high_resolution_clock::now();
        cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.nsentences) << ")\tllh=" << llh 
                << " ppl: " << exp(llh / trs) << " err: " << (trs - right) / trs << " LF: " << results["LF"] << " UF:" << results["UF"] 
                << " LP:" << results["LP"] << " LR:" << results["LR"] << " UP:" << results["UP"] << " UR:" <<results["UR"]
                << "\t[" << dev_size << " sents in " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms]" << endl;
        if (results["LF"] > best_LF) {
          cerr << "---previous best LF:" << best_LF 
               <<" saving model to " << fname << "---" << endl;
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
    vector<vector<vector<string>>> refs, hyps;
    auto t_start = std::chrono::high_resolution_clock::now();
    unsigned corpus_size = corpus.nsentencesDev;

    int miss_head = 0;

    for (unsigned sii = 0; sii < corpus_size; ++sii) {
      const vector<unsigned>& sentence=corpus.sentencesDev[sii];
      const vector<unsigned>& sentencePos=corpus.sentencesPosDev[sii]; 
      const vector<string>& sentenceUnkStr=corpus.sentencesStrDev[sii]; 
      const vector<unsigned>& actions=corpus.correct_act_sentDev[sii];
      vector<unsigned> tsentence=sentence;
      for (auto& w : tsentence)
        if (training_vocab.count(w) == 0) w = kUNK;
      double lp = 0;
      vector<unsigned> pred;
      vector<vector<string>> cand;
      vector<Expression> word_rep; // word representations
      Expression act_rep; // final action representation
      {
      ComputationGraph cg;
      pred = parser.log_prob_parser(&cg, sentence, tsentence, sentencePos, vector<unsigned>(),
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
      vector<vector<string>> ref = parser.compute_heads(sentence, actions, corpus.actions);
      vector<vector<string>> hyp = parser.compute_heads(sentence, pred, corpus.actions);
      refs.push_back(ref);
      hyps.push_back(hyp);
      if (process_headless(hyp, cand, word_rep, act_rep, parser, corpus.actions, sentence, sentencePos) > 0) {
            miss_head++;
            cerr << corpus.intToWords[sentence[0]] << corpus.intToWords[sentence[1]]<< endl;
        }
        //cerr<<"write to file" <<endl;
      output_conll(sentence, sentencePos, sentenceUnkStr, corpus.intToWords, corpus.intToPos, hyp);
      //correct_heads += compute_correct(ref, hyp, sentence.size() - 1);
      //total_heads += sentence.size() - 1;
    }
    //cerr << "miss head number: " << miss_head << endl;
    map<string, double> results = evaluate(refs, hyps, corpus.sentencesPosDev, corpus.posToInt["PU"]);
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
