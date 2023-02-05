#ifndef UTILS_HPP
#define UTILS_HPP

#include <glob.h>

#include <string>
#include <vector>

std::vector<std::string> Glob(const std::string& input_dir) {
  glob_t globbuf;
  std::vector<std::string> files;
  glob((input_dir + "*").c_str(), 0, NULL, &globbuf);
  for (int i = 0; i < globbuf.gl_pathc; i++) {
    files.push_back(globbuf.gl_pathv[i]);
  }
  globfree(&globbuf);
  return files;
}

#endif
