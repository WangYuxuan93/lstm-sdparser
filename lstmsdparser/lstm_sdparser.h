#ifndef __LTP_LSTMSDPARSER_PARSER_H__
#define __LTP_LSTMSDPARSER_PARSER_H__

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

namespace ltp {
namespace lstmsdparser {

using namespace dynet::expr;
using namespace dynet;
using namespace std;
namespace po = boost::program_options;

constexpr const char* ROOT_SYMBOL = "ROOT";
const std::string REL_NULL = "-NULL-";
const std::string REL_EXIST = "-EXIST-";

std::string StrToLower(const std::string s);

typedef struct Sizes {
	unsigned kROOT_SYMBOL;
	unsigned ACTION_SIZE;
	unsigned VOCAB_SIZE;
	unsigned POS_SIZE;
}Sizes;

typedef struct Options {
	unsigned LAYERS; // 2
	unsigned INPUT_DIM; // 100
	unsigned HIDDEN_DIM; // 200
	unsigned ACTION_DIM; // 50
	unsigned PRETRAINED_DIM; // 100
	unsigned LSTM_INPUT_DIM; // 200
	unsigned POS_DIM; // 50
	unsigned REL_DIM; // 50
  unsigned BILSTM_HIDDEN_DIM; // 100
	std::string transition_system; // "list"
  std::string dynet_seed;
  std::string dynet_mem;
	bool USE_POS; // true
  bool USE_BILSTM; // true
  bool USE_TREELSTM; // true
}Options;

static volatile bool requested_stop;

class LSTMParser {
public:
  bool DEBUG = false;
  cpyp::Corpus corpus;
  vector<unsigned> possible_actions;
	unordered_map<unsigned, vector<float>> pretrained;
	Options Opt;
	Sizes System_size;
	std::string transition_system;
  Model model;

  bool use_pretrained; // True if use pretraiend word embedding
  
  unsigned kUNK;
  set<unsigned> training_vocab; // words available in the training corpus
  set<unsigned> singletons;

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

  explicit LSTMParser();
  ~LSTMParser();

  void set_options(Options opts);
  bool load(string model_file, string training_data_file, string word_embedding_file,
                        string dev_data_file = "");

  void get_dynamic_infos();

  //bool has_path_to(int w1, int w2, const std::vector<bool>  dir_graph []);
  bool has_path_to(int w1, int w2, const std::vector<std::vector<bool>>& graph);

  bool has_path_to(int w1, int w2, const std::vector<std::vector<string>>& graph);

  vector<unsigned> get_children(unsigned id, const vector<vector<bool>> graph);

  bool IsActionForbidden(const string& a, unsigned bsize, unsigned ssize, unsigned root, const std::vector<std::vector<bool>> dir_graph,//const std::vector<bool>  dir_graph [], 
                                                const std::vector<int>& stacki, const std::vector<int>& bufferi);
  std::vector<std::vector<string>> compute_heads(const std::vector<unsigned>& sent, const std::vector<unsigned>& actions);
  std::vector<unsigned> log_prob_parser(ComputationGraph* hg,
                     const std::vector<unsigned>& raw_sent,  // raw sentence
                     const std::vector<unsigned>& sent,  // sent with oovs replaced
                     const std::vector<unsigned>& sentPos,
                     const std::vector<unsigned>& correct_actions,
                     //const std::vector<string>& setOfActions,
                     //const map<unsigned, std::string>& intToWords,
                     double *right, 
                     std::vector<std::vector<string>>& cand,
                     std::vector<Expression>* word_rep = NULL,
                     Expression * act_rep = NULL);

  int process_headless(std::vector<std::vector<string>>& hyp, std::vector<std::vector<string>>& cand, std::vector<Expression>& word_rep, 
                                    Expression& act_rep, const std::vector<unsigned>& sent, const std::vector<unsigned>& sentPos);

  void process_headless_search_all(const std::vector<unsigned>& sent, const std::vector<unsigned>& sentPos, 
                                                        const std::vector<string>& setOfActions, std::vector<Expression>& word_rep, 
                                                        Expression& act_rep, int n, int sent_len, int dir, map<int, double>* scores, 
                                                        map<int, string>* rels);

  void get_best_label(const std::vector<unsigned>& sent, const std::vector<unsigned>& sentPos, 
                                    ComputationGraph* hg, const std::vector<string>& setOfActions, 
                                    int s0, int b0, std::vector<Expression>& word_rep, Expression& act_rep, int sent_size, 
                                    int dir, double *score, string *rel);

  static void signal_callback_handler(int /* signum */);

  void train(const std::string fname, const unsigned unk_strategy, const double unk_prob);

  void predict_dev();

  void predict(std::vector<std::vector<string>> &hyp, const std::vector<std::string> & words,
                          const std::vector<std::string> & postags);

  void output_conll(const vector<unsigned>& sentence, const vector<unsigned>& pos,
                  const vector<string>& sentenceUnkStrings, 
                  //const map<unsigned, string>& intToWords, 
                  //const map<unsigned, string>& intToPos, 
                  const vector<vector<string>>& hyp);

  map<string, double> evaluate(const std::vector<std::vector<std::vector<string>>>& refs, const std::vector<std::vector<std::vector<string>>>& hyps);
};



} //  namespace lstmsdparser
} //  namespace ltp

#endif  //  end for __LTP_LSTMSDPARSER_PARSER_H__
