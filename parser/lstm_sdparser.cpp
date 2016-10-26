#include "lstm_sdparser.h"

namespace ltp {
namespace lstmsdparser {

using namespace cnn::expr;
using namespace cnn;
using namespace std;
namespace po = boost::program_options;


//struct LSTMParser {

LSTMParser::LSTMParser(): Opt({2, 100, 200, 50, 100, 200, 50, 50, "list", true}) {}

LSTMParser::~LSTMParser() {}

void LSTMParser::set_options(Options opts){
  this->Opt = opts;
}

bool LSTMParser::load(string model_file, const unordered_map<unsigned, vector<float>>& pretrained, 
                                const vector<unsigned> possible_actions, Sizes System_size){
  this->pretrained = pretrained;
  this->possible_actions = possible_actions;
  this->System_size = System_size;
  this->transition_system = Opt.transition_system;

  cerr << "setup model " << endl;

  stack_lstm = LSTMBuilder(Opt.LAYERS, Opt.LSTM_INPUT_DIM, Opt.HIDDEN_DIM, &model);
  buffer_lstm = LSTMBuilder(Opt.LAYERS, Opt.LSTM_INPUT_DIM, Opt.HIDDEN_DIM, &model);
  pass_lstm = LSTMBuilder(Opt.LAYERS, Opt.LSTM_INPUT_DIM, Opt.HIDDEN_DIM, &model);
  action_lstm = LSTMBuilder(Opt.LAYERS, Opt.ACTION_DIM, Opt.HIDDEN_DIM, &model);
  p_w = model.add_lookup_parameters(System_size.VOCAB_SIZE, {Opt.INPUT_DIM});
  p_a = model.add_lookup_parameters(System_size.ACTION_SIZE, {Opt.ACTION_DIM});
  p_r = model.add_lookup_parameters(System_size.ACTION_SIZE, {Opt.REL_DIM});
  p_pbias = model.add_parameters({Opt.HIDDEN_DIM});
  p_A = model.add_parameters({Opt.HIDDEN_DIM, Opt.HIDDEN_DIM});
  p_B = model.add_parameters({Opt.HIDDEN_DIM, Opt.HIDDEN_DIM});
  p_P = model.add_parameters({Opt.HIDDEN_DIM, Opt.HIDDEN_DIM});
  p_S = model.add_parameters({Opt.HIDDEN_DIM, Opt.HIDDEN_DIM});
  p_H = model.add_parameters({Opt.LSTM_INPUT_DIM, Opt.LSTM_INPUT_DIM});
  p_D = model.add_parameters({Opt.LSTM_INPUT_DIM, Opt.LSTM_INPUT_DIM});
  p_R = model.add_parameters({Opt.LSTM_INPUT_DIM, Opt.REL_DIM});
  p_w2l = model.add_parameters({Opt.LSTM_INPUT_DIM, Opt.INPUT_DIM});
  p_ib = model.add_parameters({Opt.LSTM_INPUT_DIM});
  p_cbias = model.add_parameters({Opt.LSTM_INPUT_DIM});
  p_p2a = model.add_parameters({System_size.ACTION_SIZE, Opt.HIDDEN_DIM});
  p_action_start = model.add_parameters({Opt.ACTION_DIM});
  p_abias = model.add_parameters({System_size.ACTION_SIZE});
  p_buffer_guard = model.add_parameters({Opt.LSTM_INPUT_DIM});
  p_stack_guard = model.add_parameters({Opt.LSTM_INPUT_DIM});
  p_pass_guard = model.add_parameters({Opt.LSTM_INPUT_DIM});
  if (Opt.USE_POS) {
    p_p = model.add_lookup_parameters(System_size.POS_SIZE, {Opt.POS_DIM});
    p_p2l = model.add_parameters({Opt.LSTM_INPUT_DIM, Opt.POS_DIM});
  }
  if (pretrained.size() > 0) {
    p_t = model.add_lookup_parameters(System_size.VOCAB_SIZE, {Opt.PRETRAINED_DIM});
    for (auto it : pretrained)
      p_t->Initialize(it.first, it.second);
    p_t2l = model.add_parameters({Opt.LSTM_INPUT_DIM, Opt.PRETRAINED_DIM});
  }else {
    p_t = nullptr;
    p_t2l = nullptr;
  }

  //this->model = model;
  if (model_file.length() > 0) {
    cerr << "loading model from " << model_file << endl;
    ifstream in(model_file.c_str());
    boost::archive::text_iarchive ia(in);
    ia >> this->model;
    cerr << "finish loading model" << endl;
  }
}

void LSTMParser::setup_system(){
  


}

bool LSTMParser::has_path_to(int w1, int w2, const vector<bool>  dir_graph []){
    //cerr << endl << w1 << " has path to " << w2 << endl;
    if (dir_graph[w1][w2])
        return true;
    for (int i = 0; i < (int)dir_graph[w1].size(); ++i){
        if (dir_graph[w1][i])
            if (has_path_to(i, w2, dir_graph))
                return true;
    }
    return false;
}

 bool LSTMParser::IsActionForbidden(const string& a, unsigned bsize, unsigned ssize, unsigned root, const vector<bool>  dir_graph [], 
                                                const vector<int>& stacki, const vector<int>& bufferi) {
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

vector<vector<string>> LSTMParser::compute_heads(const vector<unsigned>& sent, const vector<unsigned>& actions, 
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

// *** if correct_actions is empty, this runs greedy decoding ***
// returns parse actions for input sentence (in training just returns the reference)
// OOV handling: raw_sent will have the actual words
//               sent will have words replaced by appropriate UNK tokens
// this lets us use pretrained embeddings, when available, for words that were OOV in the
// parser training data
vector<unsigned> LSTMParser::log_prob_parser(ComputationGraph* hg,
                     const vector<unsigned>& raw_sent,  // raw sentence
                     const vector<unsigned>& sent,  // sent with oovs replaced
                     const vector<unsigned>& sentPos,
                     const vector<unsigned>& correct_actions,
                     const vector<string>& setOfActions,
                     const map<unsigned, std::string>& intToWords,
                     double *right, 
                     vector<vector<string>>& cand,
                     vector<Expression>* word_rep,
                     Expression * act_rep) {
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
    buffer_lstm.new_graph(*hg);
    pass_lstm.new_graph(*hg);
    action_lstm.new_graph(*hg);
    stack_lstm.start_new_sequence();
    buffer_lstm.start_new_sequence();
    pass_lstm.start_new_sequence();
    action_lstm.start_new_sequence();
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
    if (Opt.USE_POS)
      p2l = parameter(*hg, p_p2l);
    Expression t2l;
    if (p_t2l)
      t2l = parameter(*hg, p_t2l);
    Expression p2a = parameter(*hg, p_p2a);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);

    action_lstm.add_input(action_start);
    vector<Expression> buffer(sent.size() + 1);  // variables representing word embeddings (possibly including POS info)
    vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence
    // precompute buffer representation from left to right

    for (unsigned i = 0; i < sent.size(); ++i) {
      assert(sent[i] < System_size.VOCAB_SIZE);
      Expression w =lookup(*hg, p_w, sent[i]);

      vector<Expression> args = {ib, w2l, w}; // learn embeddings
      if (Opt.USE_POS) { // learn POS tag?
        Expression p = lookup(*hg, p_p, sentPos[i]);
        args.push_back(p2l);
        args.push_back(p);
      }
      if (p_t && pretrained.count(raw_sent[i])) {  // include fixed pretrained vectors?
        Expression t = const_lookup(*hg, p_t, raw_sent[i]);
        args.push_back(t2l);
        args.push_back(t);
      }
      //buffer[] = ib + w2l * w + p2l * p + t2l * t
      buffer[sent.size() - i] = rectify(affine_transform(args));
      bufferi[sent.size() - i] = i;
    }
    // dummy symbol to represent the empty buffer
    buffer[0] = parameter(*hg, p_buffer_guard);
    bufferi[0] = -999;
    for (auto& b : buffer)
      buffer_lstm.add_input(b);

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
    vector<bool> dir_graph[sent.size()]; // store the connection between words in sent
    vector<bool> v;
    for (int i = 0; i < (int)sent.size(); i++){
        v.push_back(false);
    }
    for (int i = 0; i < (int)sent.size(); i++){
        dir_graph[i] = v;
    }
    //remove stack.size() > 2 ||
    while( buffer.size() > 1) {
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
      cerr <<"]"<<endl;
        //}*/
      for (auto a: possible_actions) {
        //cerr << " " << setOfActions[a]<< " ";
        if (IsActionForbidden(setOfActions[a], buffer.size(), stack.size(), sent.size() - 1, dir_graph, stacki, bufferi))
          continue;
        //cerr << " <" << setOfActions[a] << "> ";
        current_valid_actions.push_back(a);
      }
      // p_t = pbias + S * slstm + P * plstm + B * blstm + A * almst
      Expression p_t = affine_transform({pbias, S, stack_lstm.back(), P, pass_lstm.back(), B, buffer_lstm.back(), A, action_lstm.back()});


      Expression nlp_t = rectify(p_t);
      // r_t = abias + p2a * nlp
      Expression r_t = affine_transform({abias, p2a, nlp_t});

      // adist = log_softmax(r_t, current_valid_actions)
      Expression adiste = log_softmax(r_t, current_valid_actions);
      vector<float> adist = as_vector(hg->incremental_forward());
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

        if (transition_system == "list"){
            if (ac =='N' && ac2=='S') {  // NO-SHIFT
                assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
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
                buffer_lstm.rewind_one_step();
                stacki.push_back(bufferi.back());
                bufferi.pop_back();
            } else if (ac=='N' && ac2=='R'){
                assert(stack.size() > 1);
                if (word_rep)
                    (*word_rep)[stacki.back()] = stack.back();
                stack.pop_back();
                stacki.pop_back();
                stack_lstm.rewind_one_step();
            } else if (ac=='N' && ac2=='P'){
                assert(stack.size() > 1);
                pass.push_back(stack.back());
                pass_lstm.add_input(stack.back());
                passi.push_back(stacki.back());
                stack.pop_back();
                stacki.pop_back();
                stack_lstm.rewind_one_step();
            } else if (ac=='L'){ // LEFT-REDUCE or LEFT-PASS
                assert(stack.size() > 1 && buffer.size() > 1);
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
                if (headi == sent.size() - 1) rootword = intToWords.find(sent[depi])->second;
                Expression composed = affine_transform({cbias, H, head, D, dep, R, relation});
                Expression nlcomposed = tanh(composed);
                stack_lstm.rewind_one_step();
                buffer_lstm.rewind_one_step();

                buffer_lstm.add_input(nlcomposed);
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
                assert(stack.size() > 1 && buffer.size() > 1);
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
                if (headi == sent.size() - 1) rootword = intToWords.find(sent[depi])->second;
                Expression composed = affine_transform({cbias, H, head, D, dep, R, relation});
                Expression nlcomposed = tanh(composed);

                stack_lstm.rewind_one_step();
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
                    buffer_lstm.add_input(dep);
                    buffer.push_back(dep);
                    bufferi.push_back(depi);
                }
            }
        }
        else if (transition_system == "spl"){
            //TODO
            if (ac =='N' && ac2=='S') {  // NO-SHIFT
                assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
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
                buffer_lstm.rewind_one_step();
                stacki.push_back(bufferi.back());
                bufferi.pop_back();
            } else if (ac=='N' && ac2=='P'){
                assert(stack.size() > 1);
                pass.push_back(stack.back());
                pass_lstm.add_input(stack.back());
                passi.push_back(stacki.back());
                stack.pop_back();
                stacki.pop_back();
                stack_lstm.rewind_one_step();
            } else if (ac=='L'){ // LEFT-ARC or LEFT-POP
                assert(stack.size() > 1 && buffer.size() > 1);
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
                if (headi == sent.size() - 1) rootword = intToWords.find(sent[depi])->second;
                Expression composed = affine_transform({cbias, H, head, D, dep, R, relation});
                Expression nlcomposed = tanh(composed);
                stack_lstm.rewind_one_step();
                buffer_lstm.rewind_one_step();

                buffer_lstm.add_input(nlcomposed);
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
                assert(stack.size() > 1 && buffer.size() > 1);
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
                if (headi == sent.size() - 1) rootword = intToWords.find(sent[depi])->second;
                Expression composed = affine_transform({cbias, H, head, D, dep, R, relation});
                Expression nlcomposed = tanh(composed);

                stack_lstm.rewind_one_step();
                buffer_lstm.rewind_one_step();
                pass_lstm.add_input(nlcomposed);
                pass.push_back(nlcomposed);
                passi.push_back(headi);
                buffer_lstm.add_input(dep);
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

  void LSTMParser::process_headless_search_all(const vector<unsigned>& sent, const vector<unsigned>& sentPos, 
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

  void LSTMParser::get_best_label(const vector<unsigned>& sent, const vector<unsigned>& sentPos, 
                                    ComputationGraph* hg, const vector<string>& setOfActions, 
                                    int s0, int b0, vector<Expression>& word_rep, Expression& act_rep, int sent_size, 
                                    int dir, double *score, string *rel) {
    char prefix = (dir > 0 ? 'L' : 'R');
    //init graph connecting vector
    vector<bool> dir_graph[sent_size]; // store the connection between words in sent
    vector<bool> v;
    for (int i = 0; i < sent_size; i++){
        v.push_back(false);
    }
    for (int i = 0; i < sent_size; i++){
        dir_graph[i] = v;
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
    if (Opt.USE_POS)
      p2l = parameter(*hg, p_p2l);
    Expression t2l;
    if (p_t2l)
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
    if (Opt.USE_POS) { // learn POS tag?
        Expression p = lookup(*hg, p_p, sentPos[b0]);
        args.push_back(p2l);
        args.push_back(p);
    }
    if (p_t && pretrained.count(sent[b0])) {  // include fixed pretrained vectors?
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
    if (Opt.USE_POS) { // learn POS tag?
        Expression p = lookup(*hg, p_p, sentPos[s0]);
        args.push_back(p2l);
        args.push_back(p);
    }
    if (p_t && pretrained.count(sent[s0])) {  // include fixed pretrained vectors?
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
    vector<float> adist = as_vector(hg->incremental_forward());
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

//}; //end of the struct

bool LSTMParser::has_path_to(int w1, int w2, const vector<vector<string>>& graph){
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

int LSTMParser::process_headless(vector<vector<string>>& hyp, vector<vector<string>>& cand, vector<Expression>& word_rep, 
                                    Expression& act_rep, const vector<string>& setOfActions, 
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
                process_headless_search_all(sent, sentPos, setOfActions, word_rep, act_rep, i, (int)(hyp.size()), 1, &scores, &rels);
                process_headless_search_all(sent, sentPos, setOfActions, word_rep, act_rep, i, (int)(hyp.size()), -1, &scores, &rels);
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



} //  namespace lstmsdparser
} //  namespace ltp