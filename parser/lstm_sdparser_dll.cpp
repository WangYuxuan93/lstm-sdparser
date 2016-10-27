#include "lstm_sdparser_dll.h"
#include "lstm_sdparser.h"
#include <iostream>

class __ltp_dll_lstmsdparser_wrapper : public ltp::lstmsdparser::LSTMParser {
public:
  __ltp_dll_lstmsdparser_wrapper() {}
  ~__ltp_dll_lstmsdparser_wrapper() {}

  bool load(const char * model_file, const char * training_data_file, 
            const char * word_embedding_file) {
    if (!ltp::lstmsdparser::LSTMParser::load(std::string(model_file), std::string(training_data_file), 
                                              std::string(word_embedding_file))) {
      return false;
    }
    return true;
  }

  int parse(const std::vector<std::string> & words,
            const std::vector<std::string> & postags,
            std::vector<std::vector<std::string>> & hyp) {

    //std::vector<std::vector<string>> hyp;

    ltp::lstmsdparser::LSTMParser::predict(hyp, words, postags);
    return hyp.size();
  }
};

void * lstmsdparser_create_parser(const char * model_file, const char * training_data_file, 
            const char * word_embedding_file) {
  __ltp_dll_lstmsdparser_wrapper* wrapper = new __ltp_dll_lstmsdparser_wrapper();

  if (!wrapper->load(model_file, training_data_file, word_embedding_file)) {
    delete wrapper;
    return 0;
  }
  return reinterpret_cast<void *>(wrapper);
}

int lstmsdparser_release_parser(void * parser) {
  if (!parser) {
    return -1;
  }
  delete reinterpret_cast<__ltp_dll_lstmsdparser_wrapper*>(parser);
  return 0;
}

int lstmsdparser_parse(void * parser,
                 const std::vector<std::string> & words,
                 const std::vector<std::string> & postags,
                 std::vector<std::vector<std::string>> & hyp) {
  if (words.size() != postags.size()) {
    return 0;
  }
  for (int i = 0; i < words.size(); ++ i) {
    if (words[i].empty() || postags[i].empty()) {
      return 0;
    }
  }

  __ltp_dll_lstmsdparser_wrapper* wrapper = 0;
  wrapper = reinterpret_cast<__ltp_dll_lstmsdparser_wrapper*>(parser);
  return wrapper->parse(words, postags, hyp);
}
