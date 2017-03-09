#ifndef __LTP_LSTMSDPARSER_CPYPDICT_H__
#define __LTP_LSTMSDPARSER_CPYPDICT_H__

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <functional>
#include <vector>
#include <map>
#include <string>

namespace cpyp {

class Corpus {
 //typedef std::unordered_map<std::string, unsigned, std::hash<std::string> > Map;
// typedef std::unordered_map<unsigned,std::string, std::hash<std::string> > ReverseMap;
public: 
   bool DEBUG = false;
   bool USE_SPELLING=false;
   std::string transition_system; 

   std::map<int,std::vector<unsigned>> correct_act_sent;
   std::map<int,std::vector<unsigned>> sentences;
   std::map<int,std::vector<unsigned>> sentencesPos;

   std::map<int,std::vector<unsigned>> correct_act_sentDev;
   std::map<int,std::vector<unsigned>> sentencesDev;
   std::map<int,std::vector<unsigned>> sentencesPosDev;
   std::map<int,std::vector<std::string>> sentencesStrDev;
   unsigned nsentencesDev;

   unsigned nsentences;
   unsigned nwords;
   unsigned nactions;
   unsigned npos;

   unsigned nsentencestest;
   unsigned nsentencesdev;
   int max;
   int maxPos;

   std::map<std::string, unsigned> wordsToInt;
   std::map<unsigned, std::string> intToWords;
   std::vector<std::string> actions;

   std::map<std::string, unsigned> posToInt;
   std::map<unsigned, std::string> intToPos;

   int maxChars;
   std::map<std::string, unsigned> charsToInt;
   std::map<unsigned, std::string> intToChars;

   // String literals
   static constexpr const char* UNK = "UNK";
   static constexpr const char* BAD0 = "<BAD0>";

   //! The tree type.
  typedef std::vector<std::vector<int> > tree_t;
  //! The MPC calculate result type
  typedef std::tuple<bool, int, int> mpc_result_t;

/*  std::map<unsigned,unsigned>* headsTraining;
  std::map<unsigned,std::string>* labelsTraining;

  std::map<unsigned,unsigned>*  headsParsing;
  std::map<unsigned,std::string>* labelsParsing;*/


 
 public:
  Corpus() {
    max = 0;
    maxPos = 0;
    maxChars=0; //Miguel
  }


inline unsigned UTF8Len(unsigned char x) {
  if (x < 0x80) return 1;
  else if ((x >> 5) == 0x06) return 2;
  else if ((x >> 4) == 0x0e) return 3;
  else if ((x >> 3) == 0x1e) return 4;
  else if ((x >> 2) == 0x3e) return 5;
  else if ((x >> 1) == 0x7e) return 6;
  else return 0;
}


void set_transition_system(std::string system){
  transition_system = system;
}

inline void split(const std::string& source, std::vector<std::string>& ret,
    int maxsplit=-1) {
  std::string str(source);
  int numsplit = 0;
  int len = str.size();
  size_t pos;
  for (pos = 0; pos < str.size() && (str[pos] == ' ' || str[pos] == '\t'
        || str[pos] == '\n' || str[pos] == '\r'); ++ pos);
  str = str.substr(pos);

  ret.clear();
  while (str.size() > 0) {
    pos = std::string::npos;

    for (pos = 0; pos < str.size() && (str[pos] != ' '
          && str[pos] != '\t'
          && str[pos] != '\r'
          && str[pos] != '\n'); ++ pos);

    if (pos == str.size()) {
      pos = std::string::npos;
    }

    if (maxsplit >= 0 && numsplit < maxsplit) {
      ret.push_back(str.substr(0, pos));
      ++ numsplit;
    } else if (maxsplit >= 0 && numsplit == maxsplit) {
      ret.push_back(str);
      ++ numsplit;
    } else if (maxsplit == -1) {
      ret.push_back(str.substr(0, pos));
      ++ numsplit;
    }

    if (pos == std::string::npos) {
      str = "";
    } else {
      for (; pos < str.size() && (str[pos] == ' '
            || str[pos] == '\t'
            || str[pos] == '\n'
            || str[pos] == '\r'); ++ pos);
      str = str.substr(pos);
    }
  }
}

/**
 * Return a list of words of string str, the word are separated by
 * separator.
 *
 *  @param  str         std::string     the string
 *  @param  maxsplit    std::string     the sep upperbound
 *  @return             std::vector<std::string> the words
 */
inline std::vector<std::string> split(const std::string& source, int maxsplit = -1) {
  std::vector<std::string> ret;
  split(source, ret, maxsplit);
  return ret;
}

void get_oracle_actions_calculate_orders(int root,
    const tree_t& tree,
    std::vector<int>& orders,
    int& timestamp) {
  const std::vector<int>& children = tree[root];
  if (children.size() == 0) {
    orders[root] = timestamp;
    timestamp += 1;
    return;
  }

  int i;
  for (i = 0; i < children.size() && children[i] < root; ++ i) {
    int child = children[i];
    get_oracle_actions_calculate_orders(child, tree, orders, timestamp);
  }

  orders[root] = timestamp;
  timestamp += 1;

  for (; i < children.size(); ++ i) {
    int child = children[i];
    get_oracle_actions_calculate_orders(child, tree, orders, timestamp);
  }
}

mpc_result_t get_oracle_actions_calculate_mpc(int root,
    const tree_t& tree,
    std::vector<int>& MPC) {
  const std::vector<int>& children = tree[root];
  if (children.size() == 0) {
    MPC[root] = root;
    return std::make_tuple(true, root, root);
  }

  int left = root, right = root;
  bool overall = true;

  int pivot = -1;
  for (pivot = 0; pivot < children.size() && children[pivot] < root; ++ pivot);

  for (int i = pivot - 1; i >= 0; -- i) {
    int child = children[i];
    mpc_result_t result =
      get_oracle_actions_calculate_mpc(child, tree, MPC);
    overall = overall && std::get<0>(result);
    if (std::get<0>(result) == true && std::get<2>(result) + 1 == left) {
      left = std::get<1>(result);
    } else {
      overall = false;
    }
  }

  for (int i = pivot; i < children.size(); ++ i) {
    int child = children[i];
    mpc_result_t result = get_oracle_actions_calculate_mpc(child, tree, MPC);
    overall = overall && std::get<0>(result);
    if (std::get<0>(result) == true && right + 1 == std::get<1>(result)) {
      right = std::get<2>(result);
    } else {
      overall = false;
    }
  }

  for (int i = left; i <= right; ++ i) { MPC[i] = root; }

  return std::make_tuple(overall, left, right);
}

void get_swap_oracle_actions_onestep(std::map<int, std::vector<std::pair<int, std::string>>>& graph,
    const tree_t& tree,
    std::vector<int>& heads_rec,
    std::vector<int>& sigma,
    std::vector<int>& beta,
    std::vector<std::string>& actions,
    const std::vector<int>& orders,
    const std::vector<int>& MPC) {
  //! the head will be saved in heads record after been reduced
  
  if (sigma.size() < 2) {
    actions.push_back("SHIFT");
    sigma.push_back(beta.back()); beta.pop_back();
    return;
  }

  int top0 = sigma.back();
  int top1 = sigma[sigma.size() - 2];

  //INFO_LOG("step1 %d %d %d",top1,top0,beta.back());
  if (graph[top1].back().first == top0) {
    bool all_found = true;
    for (int c: tree[top1]) { if (heads_rec[c] == -1) { all_found = false; } }
    if (all_found) {
      actions.push_back("LEFT-ARC("+graph[top1].back().second+")");
      sigma.pop_back(); sigma.back() = top0; heads_rec[top1] = top0;
      return;
    }
  }
  if (graph[top0].back().first == top1) {
    bool all_found = true;
    for (int c: tree[top0]) { if (heads_rec[c] == -1) { all_found = false; } }
    if (all_found) {
      actions.push_back("RIGHT-ARC("+graph[top0].back().second+")");
      sigma.pop_back(); heads_rec[top0] = top1;
      return;
    }
  }
  int k = beta.empty() ? -1 : beta.back();
  if ((orders[top0] < orders[top1]) &&
      (k == -1 || MPC[top0] != MPC[k])) {
    actions.push_back("SWAP");
    sigma.pop_back(); sigma.back() = top0; beta.push_back(top1);
  } else {
    actions.push_back("SHIFT");
    sigma.push_back(beta.back()); beta.pop_back();
  }
}

// return if w1 is one head of w0
bool has_head(std::map<int, std::vector<std::pair<int, std::string>>>& graph, int w0, int w1){
  if (w0 <= 0) return false;
  for (auto w : graph[w0 - 1]){
    if (w.first == w1 - 1)
      return true;
  }
  return false;
}

bool has_unfound_child(const tree_t& top_down_graph, std::vector<std::vector<bool>>& subgraph, int w){
  //std::cerr << std::endl << "has unfound child: " << w << " ";
  for (auto child : top_down_graph[w]){
    //std::cerr << child << " , ";
    if (!subgraph[child][w])
      return true;
  }
  return false;
}

//return if w has other head except the present one
bool has_other_head(std::map<int, std::vector<std::pair<int, std::string>>>& graph,
                    std::vector<std::vector<bool>>& subgraph, int w){
  int head_num = 0;
  for (auto h : subgraph[w]){
    if (h) ++head_num;
  }
  //std::cerr << "has other head: " << w << " sub: " << head_num << " gold: " << graph[w].size() << std::endl;
  if (head_num + 1 < graph[w - 1].size())
    return true;
  return false;
}

//return if w has any unfound head
bool lack_head(std::map<int, std::vector<std::pair<int, std::string>>>& graph,
                    std::vector<std::vector<bool>>& subgraph, int w){
  if (w < 0) return false;
  int head_num = 0;
  for (auto h : subgraph[w]){
    if (h) ++head_num;
  }
  if (head_num < graph[w - 1].size())
    return true;
  return false;
}

//return if w has any unfound child in stack sigma 
//!!! except the top in stack
bool has_other_child_in_stack(std::vector<std::vector<bool>>& subgraph, std::vector<int>& sigma, 
                              const tree_t& top_down_graph, int w){
  if (w < 0) return false; // w = 0 is the root
  for (auto c : top_down_graph[w]){
    if (find(sigma.begin(), sigma.end(), c) != sigma.end() 
        && c!= sigma.back() && !subgraph[c][w])
      return true;
  }
  return false;
}

//return if w has any unfound head in stack sigma 
//!!! except the top in stack
bool has_other_head_in_stack(std::vector<std::vector<bool>>& subgraph, std::vector<int>& sigma, 
                            std::map<int, std::vector<std::pair<int, std::string>>>& graph, int w){
  if (w < 0) return false; // w = -1 is the root
  for (auto h : graph[w - 1]) {
    if (find(sigma.begin(), sigma.end(), h.first + 1) != sigma.end() 
        && (h.first + 1)!= sigma.back() && !subgraph[w][h.first + 1])
      return true;
  }
  return false;
}

//return the relation between child : w0, head : w1
std::string get_arc_label(std::map<int, std::vector<std::pair<int, std::string>>>& graph,
                          int w0, int w1){
  for (auto h : graph[w0 - 1]){
    if (h.first == w1 - 1){
      return h.second;
    }
  }
  std::cerr << "ERORR in list get_arc_label!" << std::endl;
  return "-ERORR-";
}

void get_list_oracle_actions_onestep(std::map<int, std::vector<std::pair<int, std::string>>>& graph,
    const tree_t& top_down_graph,
    std::vector<int>& sigma,
    std::vector<int>& delta,
    std::vector<int>& beta,
    std::vector<std::vector<bool>>& subgraph, // subgraph[a][b] means b is the head of a
    std::vector<std::string>& actions) {
  int s0 = sigma.empty() ? -1 : sigma.back();
  int b0 = beta.empty() ? -1 : beta.back();
  //std::cerr << "s0: " << s0 << " b0: "<< b0 << std::endl;

  if (s0 > 0 && has_head(graph, s0, b0)) { // left s0 <- b0
    if ( !has_unfound_child(top_down_graph, subgraph, s0)
        && !has_other_head(graph, subgraph, s0)){
      actions.push_back("LR(" + get_arc_label(graph, s0, b0) + ")");
      sigma.pop_back(); subgraph[s0][b0] = true;
      return;
    }
    else{ //has other child or head
      actions.push_back("LP(" + get_arc_label(graph, s0, b0) + ")");
      delta.push_back(sigma.back()); sigma.pop_back(); subgraph[s0][b0] = true;
      return;
    }
  }
  else if ( s0 > 0 && has_head(graph, b0, s0)) { //right arc s0 -> b0
    if ( !has_other_child_in_stack(subgraph, sigma, top_down_graph, b0)
        && !has_other_head_in_stack(subgraph, sigma, graph, b0)){
      actions.push_back("RS(" + get_arc_label(graph, b0, s0) + ")");
      while (!delta.empty()){
        sigma.push_back(delta.back()); delta.pop_back();
      }
      sigma.push_back(beta.back()); beta.pop_back(); subgraph[b0][s0] = true;
      return;
    }
    else if (s0 > 0){
      actions.push_back("RP(" + get_arc_label(graph, b0, s0) + ")");
      delta.push_back(sigma.back()); sigma.pop_back(); subgraph[b0][s0] = true;
      return;
    }
  }
  else if (!beta.empty()
          && !has_other_child_in_stack(subgraph, sigma, top_down_graph, b0)
          && !has_other_head_in_stack(subgraph, sigma, graph, b0) 
          ){
    actions.push_back("NS");
    while (!delta.empty()){
      sigma.push_back(delta.back()); delta.pop_back();
    }
    sigma.push_back(beta.back()); beta.pop_back();
    return;
  }
  else if ( s0 > 0
          && !has_unfound_child(top_down_graph, subgraph, s0)
          && !lack_head(graph, subgraph, s0)){
    actions.push_back("NR");
    sigma.pop_back();
    return;
  }
  else if ( s0 > 0){
    actions.push_back("NP");
    delta.push_back(sigma.back()); sigma.pop_back();
    return; 
  }
  else {
    actions.push_back("-E-");
    std::cerr << "error in oracle!" << std::endl;
    return;
  }
}

void get_actions(std::map<int, std::vector<std::pair<int, std::string>>> graph,
                 std::vector<std::string>& gold_actions){
  if (transition_system == "swap"){
    int N = graph.size();
    int root = -1;
    tree_t tree(N);
    for (int i = 0; i < N; ++ i) {
      int head = graph[i].back().first;
      if (head == -1) {
        if (root != -1)
          std::cerr << "error: there should be only one root." << std::endl;
        root = i;
      } else {
        tree[head].push_back(i);
      }
    }
    //! calculate the projective order
    int timestamp = 0;//!count for the order number
    std::vector<int> orders(N, -1);
    get_oracle_actions_calculate_orders(root,tree,orders,timestamp);
    std::vector<int> MPC(N, 0);
    get_oracle_actions_calculate_mpc(root,tree,MPC);
    gold_actions.clear();
    size_t len = N;
    std::vector<int> sigma;
    std::vector<int> beta;
    std::vector<int> heads_rec(N, -1);
    //std::vector<int> output(len, -1);

    //int step = 0;
    beta.push_back(-1);
    for (int i = N - 1; i >= 0; -- i) { beta.push_back(i); }
    while (!(sigma.size() ==1 && beta.empty())) {
      get_swap_oracle_actions_onestep(graph, tree, heads_rec, sigma, beta, gold_actions,orders,MPC);
    }
  }
  else if (transition_system == "list-graph" || transition_system == "list-tree"){
    int N = graph.size();
    tree_t top_down_graph(N + 1);
    int root = -1;
    // each id is +1 from graph, so when used in graph should -1
    for (int i = 0; i < N; ++ i) {
      for (auto n: graph[i]){
        int head = n.first + 1;
        if (head == -1) {
          if (root != -1)
            std::cerr << "error: there should be only one root." << std::endl;
          root = i + 1;
        } else {
          top_down_graph[head].push_back(i + 1);
        }
      }
    }
    std::vector<int> sigma;
    std::vector<int> beta;
    std::vector<int> delta; // for pass action
    std::vector<bool> v(N + 1, false);
    std::vector<std::vector<bool>> subgraph;
    for (int i = 0; i < N + 1; ++i) subgraph.push_back(v);
    beta.push_back(0);
    for (int i = N; i >= 1; -- i) { beta.push_back(i); }
    while (!beta.empty()) {
      get_list_oracle_actions_onestep(graph, top_down_graph, sigma, delta, beta, subgraph, gold_actions);
      /*std::cerr << gold_actions.back() << std::endl;
      std::cerr << "stack: ";
      for (auto i : sigma) std::cerr << i << " , ";
      std::cerr << "pass: ";
      for (auto i : delta) std::cerr << i << " , ";
      std::cerr << "buffer: ";
      for (auto i : beta) std::cerr << i <<" , ";
      std::cerr << std::endl;*/
    }
    //gold_actions.push_back("LR(Root)"); 
    //gold_actions.push_back("NS"); 
  }
}

inline void load_conll_file(std::string file){
  std::ifstream actionsFile(file);
  //correct_act_sent=new vector<vector<unsigned>>();
  if (!actionsFile){
    std::cerr << "File does not exist!" << std::endl;
  }
  std::string lineS;

  int sentence=0;
  wordsToInt[Corpus::BAD0] = 0;
  intToWords[0] = Corpus::BAD0;
  wordsToInt[Corpus::UNK] = 1; // unknown symbol
  intToWords[1] = Corpus::UNK;
  wordsToInt["ROOT"] = 2; // root
  intToWords[2] = "ROOT";
  posToInt["ROOT"] = 1; // root
  intToPos[1] = "ROOT";
  assert(max == 0);
  assert(maxPos == 0);
  max=3;
  maxPos=2;
  
  charsToInt[BAD0]=1;
  intToChars[1]="BAD0";
  maxChars=1;
  
  std::vector<unsigned> current_sent;
  std::vector<unsigned> current_sent_pos;
  bool is_tree = true;
  int ngraph = 0;
  std::map<int, std::vector<std::pair<int, std::string>>> graph;
  while (getline(actionsFile, lineS)){
    //istringstream iss(line);
    //string lineS;
    //iss>>lineS;
    ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
    ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
    if (lineS.empty()) {
      /*if (is_tree) std::cerr << "is tree" << std::endl;
      for (int j = 0; j < graph.size(); ++j){
        std::cerr << j << "\t" << intToWords[current_sent[j]] << "\t" << intToPos[current_sent_pos[j]]
        << "\t" << graph[j].back().first << "\t" << graph[j].back().second << std::endl;
      }*/

      std::vector<std::string> gold_acts;
      get_actions(graph, gold_acts);
      bool found=false;
      //std::cerr << std::endl;
      for (auto g: gold_acts){
        //std::cerr << g << std::endl;
        int i = 0;
        found=false;
        for (auto a: actions) {
          if (a==g) {
            std::vector<unsigned> a=correct_act_sent[sentence];
            a.push_back(i);
            correct_act_sent[sentence]=a;
            found=true;
            break;
          }
          ++i;
        }
        if (!found) {
          actions.push_back(g);
          std::vector<unsigned> a=correct_act_sent[sentence];
          a.push_back(actions.size()-1);
          correct_act_sent[sentence]=a;
        }
      }
      current_sent.push_back(wordsToInt["ROOT"]);
      current_sent_pos.push_back(posToInt["ROOT"]);
      sentences[sentence] = current_sent;
      sentencesPos[sentence] = current_sent_pos;    
      sentence++;
      nsentences = sentence;
      
      current_sent.clear();
      current_sent_pos.clear();
      graph.clear();
      if (!is_tree) ++ngraph;
      is_tree = true;
    } else {
      //stack and buffer, for now, leave it like this.
      // one line in each sentence may look like:
      // 5  American  american  ADJ JJ  Degree=Pos  6 amod  _ _
      // read the every line
      std::vector<std::string> items = split(lineS);
      unsigned id = std::atoi(items[0].c_str()) - 1;
      std::string word = items[1];
      std::string pos = items[3];
      unsigned head = std::atoi(items[6].c_str()) - 1;
      std::string rel = items[7];
      if (graph[id].size() > 0) is_tree = false;
      graph[id].push_back(std::make_pair(head, rel));
      // new POS tag
      if (posToInt[pos] == 0) {
        posToInt[pos] = maxPos;
        intToPos[maxPos] = pos;
        npos = maxPos;
        maxPos++;
      }
      // new word
      if (wordsToInt[word] == 0) {
        wordsToInt[word] = max;
        intToWords[max] = word;
        nwords = max;
        max++;

        unsigned j = 0;
        while(j < word.length()) {
          std::string wj = "";
          for (unsigned h = j; h < j + UTF8Len(word[j]); h++) {
            wj += word[h];
          }
          if (charsToInt[wj] == 0) {
            charsToInt[wj] = maxChars;
            intToChars[maxChars] = wj;
            maxChars++;
          }
            j += UTF8Len(word[j]);
          }
        }
        current_sent.push_back(wordsToInt[word]);
        current_sent_pos.push_back(posToInt[pos]);
    }
  }


  // Add the last sentence.
  if (current_sent.size() > 0) {
    std::vector<std::string> gold_acts;
    get_actions(graph, gold_acts);
    bool found=false;
    for (auto g: gold_acts){
      //std::cerr << g << std::endl;
      int i = 0;
      found=false;
      for (auto a: actions) {
        if (a==g) {
          std::vector<unsigned> a=correct_act_sent[sentence];
          a.push_back(i);
          correct_act_sent[sentence]=a;
          found=true;
          break;
        }
        ++i;
      }
      if (!found) {
        actions.push_back(g);
        std::vector<unsigned> a=correct_act_sent[sentence];
        a.push_back(actions.size()-1);
        correct_act_sent[sentence]=a;
      }
    }
    current_sent.push_back(wordsToInt["ROOT"]);
    current_sent_pos.push_back(posToInt["ROOT"]);
    sentences[sentence] = current_sent;
    sentencesPos[sentence] = current_sent_pos;    
    sentence++;
    nsentences = sentence;

    /*if (is_tree) std::cerr << "is tree" << std::endl;
    for (int j = 0; j < graph.size(); ++j){
      std::cerr << j << "\t" << intToWords[current_sent[j]] << "\t" << intToPos[current_sent_pos[j]]
      << "\t" << graph[j].back().first << "\t" << graph[j].back().second << std::endl;
    }*/

    if (!is_tree) ++ngraph;
  }
      
  actionsFile.close();
  std::cerr << "tree / total = " << nsentences - ngraph << " / " << nsentences << std::endl;
  if (DEBUG){
    std::cerr<<"done"<<"\n";
    for (auto a: actions)
      std::cerr<<a<<"\n";
  }
  nactions=actions.size();
  if (DEBUG){
    std::cerr<<"nactions:"<<nactions<<"\n";
    std::cerr<<"nwords:"<<nwords<<"\n";
    for (unsigned i=0;i<npos;i++)
      std::cerr<<i<<":"<<intToPos[i]<<"\n";
  }
  nactions=actions.size();
}

inline void load_conll_fileDev(std::string file){
  std::ifstream actionsFile(file);
  //correct_act_sent=new vector<vector<unsigned>>();
  if (!actionsFile){
    std::cerr << "File does not exist!" << std::endl;
  }
  std::string lineS;

  assert(maxPos > 1);
  assert(max > 3);
  int sentence=0;
  std::vector<unsigned> current_sent;
  std::vector<unsigned> current_sent_pos;
  std::vector<std::string> current_sent_str;
  bool is_tree = true;
  int ngraph = 0;
  std::map<int, std::vector<std::pair<int, std::string>>> graph;
  while (getline(actionsFile, lineS)){
    //istringstream iss(line);
    //string lineS;
    //iss>>lineS;
    ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
    ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
    if (lineS.empty()) {
      std::vector<std::string> gold_acts;
      get_actions(graph, gold_acts);
      bool found=false;
      for (auto g: gold_acts){
        //std::cerr << g << std::endl;
        auto actionIter = std::find(actions.begin(), actions.end(), g);
        if (actionIter != actions.end()) {
          unsigned actionIndex = std::distance(actions.begin(), actionIter);
          correct_act_sentDev[sentence].push_back(actionIndex);
        } else {
          // new action
          actions.push_back(g);
          unsigned actionIndex = actions.size() - 1;
          correct_act_sentDev[sentence].push_back(actionIndex);
        }
      }
      current_sent.push_back(wordsToInt["ROOT"]);
      current_sent_pos.push_back(posToInt["ROOT"]);
      current_sent_str.push_back("");
      sentencesDev[sentence] = current_sent;
      sentencesPosDev[sentence] = current_sent_pos;
      sentencesStrDev[sentence] = current_sent_str;    
      sentence++;
      nsentencesDev = sentence;

      /*if (is_tree) std::cerr << "is tree" << std::endl;
      for (int j = 0; j < graph.size(); ++j){
        std::cerr << j << "\t" << intToWords[current_sent[j]] << "\t" << intToPos[current_sent_pos[j]]
        << "\t" << graph[j].back().first << "\t" << graph[j].back().second << std::endl;
      }*/
      
      current_sent.clear();
      current_sent_pos.clear();
      current_sent_str.clear();
      graph.clear();
      if (!is_tree) ++ngraph;
      is_tree = true;
    } else {
      //stack and buffer, for now, leave it like this.
      // one line in each sentence may look like:
      // 5  American  american  ADJ JJ  Degree=Pos  6 amod  _ _
      // read the every line
      std::vector<std::string> items = split(lineS);
      unsigned id = std::atoi(items[0].c_str()) - 1;
      std::string word = items[1];
      std::string pos = items[3];
      unsigned head = std::atoi(items[6].c_str()) - 1;
      std::string rel = items[7];
      if (graph[id].size() > 0) is_tree = false;
      graph[id].push_back(std::make_pair(head, rel));
      // new POS tag
      if (posToInt[pos] == 0) {
        posToInt[pos] = maxPos;
        intToPos[maxPos] = pos;
        npos = maxPos;
        maxPos++;
      }
      // add an empty string for any token except OOVs (it is easy to 
      // recover the surface form of non-OOV using intToWords(id)).
      current_sent_str.push_back("");
      // OOV word
      if (wordsToInt[word] == 0) {
        if (USE_SPELLING) {
          max = nwords + 1;
          wordsToInt[word] = max;
          intToWords[max] = word;
          nwords = max;
        } else {
          // save the surface form of this OOV before overwriting it.
          current_sent_str[current_sent_str.size()-1] = word;
          word = Corpus::UNK;
        }
      }
      current_sent.push_back(wordsToInt[word]);
      current_sent_pos.push_back(posToInt[pos]);
    }
  }

  // Add the last sentence.
  if (current_sent.size() > 0) {
    std::vector<std::string> gold_acts;
    get_actions(graph, gold_acts);
    bool found=false;
    for (auto g: gold_acts){
      //std::cerr << g << std::endl;
      auto actionIter = std::find(actions.begin(), actions.end(), g);
      if (actionIter != actions.end()) {
        unsigned actionIndex = std::distance(actions.begin(), actionIter);
        correct_act_sentDev[sentence].push_back(actionIndex);
      } else {
        // new action
        actions.push_back(g);
        unsigned actionIndex = actions.size() - 1;
        correct_act_sentDev[sentence].push_back(actionIndex);
      }
    }
    current_sent.push_back(wordsToInt["ROOT"]);
    current_sent_pos.push_back(posToInt["ROOT"]);
    current_sent_str.push_back("");
    sentencesDev[sentence] = current_sent;
    sentencesPosDev[sentence] = current_sent_pos;
    sentencesStrDev[sentence] = current_sent_str;    
    sentence++;
    nsentencesDev = sentence;
    if (!is_tree) ++ngraph;
    /*if (is_tree) std::cerr << "is tree" << std::endl;
      for (int j = 0; j < graph.size(); ++j){
        std::cerr << j << "\t" << intToWords[current_sent[j]] << "\t" << intToPos[current_sent_pos[j]]
        << "\t" << graph[j].back().first << "\t" << graph[j].back().second << std::endl;
      }*/
  }
      
  actionsFile.close();
  std::cerr << "tree / total = " << nsentences - ngraph << " / " << nsentences << std::endl;
  if (DEBUG){
    std::cerr<<"done"<<"\n";
    for (auto a: actions)
      std::cerr<<a<<"\n";
  }
  nactions=actions.size();
  if (DEBUG){
    std::cerr<<"nactions:"<<nactions<<"\n";
    std::cerr<<"nwords:"<<nwords<<"\n";
    for (unsigned i=0;i<npos;i++)
      std::cerr<<i<<":"<<intToPos[i]<<"\n";
  }
  nactions=actions.size();
}

inline void load_correct_actions(std::string file){
	
  std::ifstream actionsFile(file);
  //correct_act_sent=new vector<vector<unsigned>>();
  if (!actionsFile){
    std::cerr << "File does not exist!" << std::endl;
  }
  std::string lineS;
	
  int count=-1;
  int sentence=-1;
  bool initial=false;
  bool first=true;
  wordsToInt[Corpus::BAD0] = 0;
  intToWords[0] = Corpus::BAD0;
  wordsToInt[Corpus::UNK] = 1; // unknown symbol
  intToWords[1] = Corpus::UNK;
  assert(max == 0);
  assert(maxPos == 0);
  max=2;
  maxPos=1;
  
  charsToInt[BAD0]=1;
  intToChars[1]="BAD0";
  maxChars=1;
  
	std::vector<unsigned> current_sent;
  std::vector<unsigned> current_sent_pos;
  while (getline(actionsFile, lineS)){
    //istringstream iss(line);
    //string lineS;
 		//iss>>lineS;
    ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
    ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
		if (lineS.empty()) {
			count = 0;
			if (!first) {
				sentences[sentence] = current_sent;
				sentencesPos[sentence] = current_sent_pos;
      }
      
			sentence++;
			nsentences = sentence;
      
			initial = true;
                   current_sent.clear();
			current_sent_pos.clear();
		} else if (count == 0) {
			first = false;
			//stack and buffer, for now, leave it like this.
			count = 1;
			if (initial) {
        // the initial line in each sentence may look like:
        // [][][the-det, cat-noun, is-verb, on-adp, the-det, mat-noun, ,-punct, ROOT-ROOT]
        // first, get rid of the square brackets.
        if (transition_system == "swap")
          lineS = lineS.substr(3, lineS.size() - 4); // 5, 6 for list-based , 3, 4 for swap
        else
          lineS = lineS.substr(5, lineS.size() - 6);
        // read the initial line, token by token "the-det," "cat-noun," ...
        std::istringstream iss(lineS);
        do {
          std::string word;
          iss >> word;
          if (word.size() == 0) { continue; }
          // remove the trailing comma if need be.
          if (word[word.size() - 1] == ',') { 
            word = word.substr(0, word.size() - 1);
          }
          // split the string (at '-') into word and POS tag.
          size_t posIndex = word.rfind('-');
          if (posIndex == std::string::npos) {
            std::cerr << "cant find the dash in '" << word << "'" << std::endl;
          }
          assert(posIndex != std::string::npos);
          std::string pos = word.substr(posIndex + 1);
          word = word.substr(0, posIndex);
          // new POS tag
          if (posToInt[pos] == 0) {
            posToInt[pos] = maxPos;
            intToPos[maxPos] = pos;
            npos = maxPos;
            maxPos++;
          }

          // new word
          if (wordsToInt[word] == 0) {
            wordsToInt[word] = max;
            intToWords[max] = word;
            nwords = max;
            max++;

            unsigned j = 0;
            while(j < word.length()) {
              std::string wj = "";
              for (unsigned h = j; h < j + UTF8Len(word[j]); h++) {
                wj += word[h];
              }
              if (charsToInt[wj] == 0) {
                charsToInt[wj] = maxChars;
                intToChars[maxChars] = wj;
                maxChars++;
              }
              j += UTF8Len(word[j]);
            }
          }
        
          current_sent.push_back(wordsToInt[word]);
          current_sent_pos.push_back(posToInt[pos]);
        } while(iss);
			}
			initial=false;
		}
		else if (count==1){
			int i=0;
			bool found=false;
			for (auto a: actions) {
				if (a==lineS) {
					std::vector<unsigned> a=correct_act_sent[sentence];
	                                a.push_back(i);
        	                        correct_act_sent[sentence]=a;
					found=true;
				}
				i++;
			}
			if (!found) {
				actions.push_back(lineS);
				std::vector<unsigned> a=correct_act_sent[sentence];
				a.push_back(actions.size()-1);
				correct_act_sent[sentence]=a;
			}
			count=0;
		}
	}

  // Add the last sentence.
  if (current_sent.size() > 0) {
    sentences[sentence] = current_sent;
    sentencesPos[sentence] = current_sent_pos;
    sentence++;
    nsentences = sentence;
  }
      
  actionsFile.close();
/*	std::string oov="oov";
	posToInt[oov]=maxPos;
        intToPos[maxPos]=oov;
        npos=maxPos;
        maxPos++;
        wordsToInt[oov]=max;
        intToWords[max]=oov;
        nwords=max;
        max++;*/
  if (DEBUG){
	  std::cerr<<"done"<<"\n";
	  for (auto a: actions) {
		  std::cerr<<a<<"\n";
	  }
  }
	nactions=actions.size();
  if (DEBUG){
	std::cerr<<"nactions:"<<nactions<<"\n";
        std::cerr<<"nwords:"<<nwords<<"\n";
	for (unsigned i=0;i<npos;i++){
                std::cerr<<i<<":"<<intToPos[i]<<"\n";
        }
  }
	nactions=actions.size();
	
}

inline unsigned get_or_add_word(const std::string& word) {
  unsigned& id = wordsToInt[word];
  if (id == 0) {
    id = max;
    ++max;
    intToWords[id] = word;
    nwords = max;
  }
  return id;
}

inline void load_correct_actionsDev(std::string file) {
  std::ifstream actionsFile(file);
  if (!actionsFile){
    std::cerr << "File does not exist!" << std::endl;
  }
  std::string lineS;

  assert(maxPos > 1);
  assert(max > 3);
  int count = -1;
  int sentence = -1;
  bool initial = false;
  bool first = true;
  std::vector<unsigned> current_sent;
  std::vector<unsigned> current_sent_pos;
  std::vector<std::string> current_sent_str;
  while (getline(actionsFile, lineS)) {
    ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
    ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
    if (lineS.empty()) {
      // an empty line marks the end of a sentence.
      count = 0;
      if (!first) {
        sentencesDev[sentence] = current_sent;
        sentencesPosDev[sentence] = current_sent_pos;
        sentencesStrDev[sentence] = current_sent_str;
      }
      
      sentence++;
      nsentencesDev = sentence;
      
      initial = true;
      current_sent.clear();
      current_sent_pos.clear();
      current_sent_str.clear(); 
    } else if (count == 0) {
      first = false;
      //stack and buffer, for now, leave it like this.
      count = 1;
      if (initial) {
        // the initial line in each sentence may look like:
        // [][the-det, cat-noun, is-verb, on-adp, the-det, mat-noun, ,-punct, ROOT-ROOT]
        // first, get rid of the square brackets.
        if (transition_system == "swap")
          lineS = lineS.substr(3, lineS.size() - 4); // 5, 6 for list-based , 3, 4 for swap
        else
          lineS = lineS.substr(5, lineS.size() - 6);
        // read the initial line, token by token "the-det," "cat-noun," ...
        std::istringstream iss(lineS);  
	do {
          std::string word;
          iss >> word;
          if (word.size() == 0) { continue; }
          // remove the trailing comma if need be.
          if (word[word.size() - 1] == ',') { 
            word = word.substr(0, word.size() - 1);
          }
          // split the string (at '-') into word and POS tag.
          size_t posIndex = word.rfind('-');
          assert(posIndex != std::string::npos);
          std::string pos = word.substr(posIndex + 1);
          word = word.substr(0, posIndex);
          // new POS tag
          if (posToInt[pos] == 0) {
            posToInt[pos] = maxPos;
            intToPos[maxPos] = pos;
            npos = maxPos;
            maxPos++;
          }
          // add an empty string for any token except OOVs (it is easy to 
          // recover the surface form of non-OOV using intToWords(id)).
          current_sent_str.push_back("");
          // OOV word
          if (wordsToInt[word] == 0) {
            if (USE_SPELLING) {
              max = nwords + 1;
              //std::cerr<< "max:" << max << "\n";
              wordsToInt[word] = max;
              intToWords[max] = word;
              nwords = max;
            } else {
              // save the surface form of this OOV before overwriting it.
              current_sent_str[current_sent_str.size()-1] = word;
              word = Corpus::UNK;
            }
          }
          current_sent.push_back(wordsToInt[word]);
          current_sent_pos.push_back(posToInt[pos]);
        } while(iss);
      }
      initial = false;
    } else if (count == 1) {
      auto actionIter = std::find(actions.begin(), actions.end(), lineS);
      if (actionIter != actions.end()) {
        unsigned actionIndex = std::distance(actions.begin(), actionIter);
        correct_act_sentDev[sentence].push_back(actionIndex);
      } else {
        // TODO: right now, new actions which haven't been observed in training
        // are not added to correct_act_sentDev. This may be a problem if the
        // training data is little.
          // new action
          actions.push_back(lineS);
          unsigned actionIndex = actions.size() - 1;
          correct_act_sentDev[sentence].push_back(actionIndex);
      }
      count=0;
    }
  }

  // Add the last sentence.
  if (current_sent.size() > 0) {
    sentencesDev[sentence] = current_sent;
    sentencesPosDev[sentence] = current_sent_pos;
    sentencesStrDev[sentence] = current_sent_str;
    sentence++;
    nsentencesDev = sentence;
  }
  
  actionsFile.close();
}

void ReplaceStringInPlace(std::string& subject, const std::string& search,
                          const std::string& replace) {
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos) {
         subject.replace(pos, search.length(), replace);
         pos += replace.length();
    }
}


/*  inline unsigned max() const { return words_.size(); }
  inline unsigned size() const { return words_.size(); }
  inline unsigned count(const std::string& word) const { return d_.count(word); }*/

/*  static bool is_ws(char x) {
    return (x == ' ' || x == '\t');
  }

  inline void ConvertWhitespaceDelimitedLine(const std::string& line, std::vector<unsigned>* out) {
    size_t cur = 0;
    size_t last = 0;
    int state = 0;
    out->clear();
    while(cur < line.size()) {
      if (is_ws(line[cur++])) {
        if (state == 0) continue;
        out->push_back(Convert(line.substr(last, cur - last - 1)));
        state = 0;
      } else {
        if (state == 1) continue;
        last = cur - 1;
        state = 1;
      }
    }
    if (state == 1)
      out->push_back(Convert(line.substr(last, cur - last)));
  }

  inline unsigned Convert(const std::string& word, bool frozen = false) {
    Map::iterator i = d_.find(word);
    if (i == d_.end()) {
      if (frozen)
        return 0;
      words_.push_back(word);
      d_[word] = words_.size();
      return words_.size();
    } else {
      return i->second;
    }
  }

  inline const std::string& Convert(const unsigned id) const {
    if (id == 0) return b0_;
    return words_[id-1];
  }
  template<class Archive> void serialize(Archive& ar, const unsigned int version) {
    ar & b0_;
    ar & words_;
    ar & d_;
  }
 private:
  std::string b0_;
  std::vector<std::string> words_;
  Map d_;*/
};

/*void ReadFromFile(const std::string& filename,
                  Corpus* d,
                  std::vector<std::vector<unsigned> >* src,
                  std::set<unsigned>* src_vocab) {
  std::cerr << "Reading from " << filename << std::endl;
  std::ifstream in(filename);
  assert(in);
  std::string line;
  int lc = 0;
  while(getline(in, line)) {
    ++lc;
    src->push_back(std::vector<unsigned>());
    d->ConvertWhitespaceDelimitedLine(line, &src->back());
    for (unsigned i = 0; i < src->back().size(); ++i) src_vocab->insert(src->back()[i]);
  }
}

void ReadParallelCorpusFromFile(const std::string& filename,
                                Corpus* d,
                                std::vector<std::vector<unsigned> >* src,
                                std::vector<std::vector<unsigned> >* trg,
                                std::set<unsigned>* src_vocab,
                                std::set<unsigned>* trg_vocab) {
  src->clear();
  trg->clear();
  std::cerr << "Reading from " << filename << std::endl;
  std::ifstream in(filename);
  assert(in);
  std::string line;
  int lc = 0;
  std::vector<unsigned> v;
  const unsigned kDELIM = d->Convert("|||");
  while(getline(in, line)) {
    ++lc;
    src->push_back(std::vector<unsigned>());
    trg->push_back(std::vector<unsigned>());
    d->ConvertWhitespaceDelimitedLine(line, &v);
    unsigned j = 0;
    while(j < v.size() && v[j] != kDELIM) {
      src->back().push_back(v[j]);
      src_vocab->insert(v[j]);
      ++j;
    }
    if (j >= v.size()) {
      std::cerr << "Malformed input in parallel corpus: " << filename << ":" << lc << std::endl;
      abort();
    }
    ++j;
    while(j < v.size()) {
      trg->back().push_back(v[j]);
      trg_vocab->insert(v[j]);
      ++j;
    }
  }
}*/

} // namespace

#endif // __LTP_LSTMSDPARSER_CPYPDICT_H__
