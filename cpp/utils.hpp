#ifndef UTILS_HPP
#define UTILS_HPP

#include <glob.h>

#include <string>
#include <vector>

std::vector<std::string> Glob(const std::string& input_dir) {
  glob_t buffer;
  std::vector<std::string> files;
  glob((input_dir + "*").c_str(), 0, NULL, &buffer);
  for (size_t i = 0; i < buffer.gl_pathc; i++) {
    files.push_back(buffer.gl_pathv[i]);
  }
  globfree(&buffer);
  return files;
}

#endif
