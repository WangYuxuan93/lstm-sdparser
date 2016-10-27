#include <iostream>
#include <vector>

#include "lstm_sdparser_dll.h"

#define EXECUTABLE "lstm_par"
#define DESCRIPTION "The console application for dependency parsing."

int main(int argc, char * argv[]) {
  if (argc < 4) {
    std::cerr << "usage: ./lstm_par [model] [training_data] [embedding_file]" << std::endl;
    return -1;
  }

  void * engine = lstmsdparser_create_parser(argv[1], argv[2], argv[3]);
  if (!engine) {
    return -1;
  }

  std::vector<std::vector<std::string>> hyp;
  std::string word[]={"我","是","中国","学生","ROOT"}; // id : 22 146 296 114 21
  size_t w_count=sizeof(word)/sizeof(std::string);
  std::string pos[]={"NN","VE","JJ","NN","ROOT"};
  size_t p_count=sizeof(word)/sizeof(std::string);
  std::vector<std::string> words(word,word+w_count);
  std::vector<std::string> postags(pos,pos+p_count);

  lstmsdparser_parse(engine, words, postags, hyp);

  for (int i = 0; i < hyp.size(); i++){
      for (int j = 0; j < hyp.size(); j++)
        std::cerr << hyp[i][j] << " ";
      std::cerr << std::endl;
  }

  lstmsdparser_release_parser(engine);
  return 0;
}

