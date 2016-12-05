#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>
#include <chrono>

#include "boost/program_options.hpp"
#include "config.h"
#include "lstm_sdparser_dll.h"

#define EXECUTABLE "lstm_par_cmdline"
#define DESCRIPTION "The console application for semantic dependency graph parsing."

using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::variables_map;
using boost::program_options::store;
using boost::program_options::parse_command_line;

void split(const std::string& src, const std::string& separator, std::vector<std::string>& dest)
{
     std::string str = src;
     std::string substring;
     std::string::size_type start = 0, index;
 
     do
     {
         index = str.find_first_of(separator,start);
         if (index != std::string::npos)
         {    
             substring = str.substr(start,index-start);
             dest.push_back(substring);
            start = str.find_first_not_of(separator,index);
             if (start == std::string::npos) return;
         }
     }while(index != std::string::npos);
      
     //the last token
     substring = str.substr(start);
     dest.push_back(substring);
}

void output_conll(std::vector<std::string> words, std::vector<std::string> postags,
                    std::vector<std::vector<std::string>> hyp){
    std::string str = "";
    std::string full = "";
    for(int i = 0; i < words.size(); i++){
      str = "";
      str += std::to_string(i+1) + "\t" + words[i] + "\t" + words[i] + "\t" + postags[i] + "\t" + postags[i] + "\t_\t";
      for (int j = 0; j < hyp[i].size(); j++){
        if (hyp[i][j] != "-NULL-"){
          full = str + std::to_string(j+1) + "\t" + hyp[i][j] + "\t_\t_";
          std::cout << full << std::endl;
        }
      }
    }
    std::cout << std::endl;
}

int main(int argc, char * argv[]) {
  std::string usage = EXECUTABLE " in LTP " LTP_VERSION " - " LTP_COPYRIGHT "\n";
  usage += DESCRIPTION "\n\n";
  usage += "usage: ./" EXECUTABLE " <options>\n\n";
  usage += "options";

  options_description optparser = options_description(usage);
  optparser.add_options()
    ("model-directory", value<std::string>(), "The directory of model folder [default=ltp_data/semparser/].")
    ("input", value<std::string>(), "The path to the input file. "
     "Input data should contain one sentence each line. "
     "Words should be separated by space with POS tag appended by "
     "'_' (e.g. \"w1_p1 w2_p2 w3_p3 w4_p4\").")
    ("help,h", "Show help information");

  if (argc == 1) {
    std::cerr << optparser << std::endl;
    return 1;
  }

  variables_map vm;
  store(parse_command_line(argc, argv, optparser), vm);

  std::string input = "";
  if (vm.count("input")) { input = vm["input"].as<std::string>(); }

  std::string model_directory = "ltp_data/semparser/";
  if (vm.count("model-directory")) {
    model_directory = vm["model-directory"].as<std::string>();
  }
  /*
  if (argc < 3) {
    std::cerr << "usage: ./lstm_par [data directory] [input file]" << std::endl;
    return -1;
  }*/

  void * engine = lstmsdparser_create_parser(model_directory.c_str());
  if (!engine) {
    std::cerr << "fail to init parser" << std::endl;
    return -1;
  }
  std::cerr << "finish loading model" << std::endl;

  std::ifstream ifs(input.c_str());

  std::string sentence = "";

  auto t_start = std::chrono::high_resolution_clock::now();
  int num = 0;
  while (!ifs.eof()) {
      std::getline(ifs, sentence,'\n');
      if (!sentence.length()) break;
      num ++;
      std::vector<std::string> sent;
      split(sentence, "\t", sent);
      std::vector<std::string> words;
      std::vector<std::string> postags;
      for (int i = 0; i < sent.size(); i++){
        int idx = sent[i].find_first_of('_', 0);
        words.push_back(sent[i].substr(0, idx));
        postags.push_back(sent[i].substr(idx + 1));
      }
      std::vector<std::vector<std::string>> hyp;
      lstmsdparser_parse(engine, words, postags, hyp);
      output_conll(words, postags, hyp);
  }

  auto t_end = std::chrono::high_resolution_clock::now();

  std::cerr << "Processed " << num << " sents in " << std::chrono::duration<double, std::milli>(t_end-t_start).count() 
              << " ms" << std::endl;

/*
  std::vector<std::vector<std::string>> hyp;
  std::string word[]={"我","是","中国","学生"}; // id : 22 146 296 114 21
  size_t w_count=sizeof(word)/sizeof(std::string);
  std::string pos[]={"r","v","ns","n"};
  size_t p_count=sizeof(word)/sizeof(std::string);
  std::vector<std::string> words(word,word+w_count);
  std::vector<std::string> postags(pos,pos+p_count);
  */

  //lstmsdparser_parse(engine, words, postags, hyp);

  lstmsdparser_release_parser(engine);
  return 0;
}

