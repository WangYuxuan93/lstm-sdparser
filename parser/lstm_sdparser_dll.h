#ifndef __LTP_LSTMSDPARSER_DLL_H__
#define __LTP_LSTMSDPARSER_DLL_H__

#include <iostream>
#include <vector>

#define LSTMSDPARSER_DLL_API
#define LSTMSDPARSER_DLL_API_EXPORT

#if defined(_MSC_VER)
#undef LSTMSDPARSER_DLL_API
#ifdef LSTMSDPARSER_DLL_API_EXPORT
    #define LSTMSDPARSER_DLL_API extern "C" _declspec(dllexport)
#else
    #define LSTMSDPARSER_DLL_API extern "C" _deslspec(dllimport)
    #pragma comment(lib, "lstm_sdparser.lib") //change the Makefile
#endif // end for PARSER_DLL_API
#endif // end for _WIN32

/*
 * create a new postagger
 *
 *  @param[in] path the path of the model
 *  @return void * the pointer to the segmentor
 */
LSTMSDPARSER_DLL_API void * lstmsdparser_create_parser(const char * model_file, const char * training_data_file, 
            const char * word_embedding_file);

/*
 * release the postagger resources
 *
 *  @param[in]  segmentor   the segmentor
 *  @return     int         i don't know
 */
LSTMSDPARSER_DLL_API int lstmsdparser_release_parser(void * parser);

/*
 * run postag given the postagger on the input words
 *
 *  @param[in]  words       the string to be segmented
 *  @param[out] tags        the words of the input line
 *  @return     int         the number of word tokens, if input arguments
 *                          are not legal, return 0
 */
LSTMSDPARSER_DLL_API int lstmsdparser_parse(void * parser,
                 const std::vector<std::string> & words,
                 const std::vector<std::string> & postags,
                 std::vector<std::vector<std::string>> & hyp);

#endif  //  end for __LTP_LSTMSDPARSER_DLL_H__
