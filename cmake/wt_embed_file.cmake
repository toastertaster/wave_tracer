# Embeds a binary file into a C++ source file:
#   cmake -DINPUT=<file> -DOUTPUT=<file.cpp> -DNAME=<symbol> -P wt_embed_file.cmake
# defines `extern const unsigned char NAME[]` and `extern const std::size_t NAME_size`.

file(READ "${INPUT}" hex HEX)
string(LENGTH "${hex}" hex_length)
math(EXPR size "${hex_length} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
file(WRITE "${OUTPUT}"
    "// Generated from ${INPUT} by wt_embed_file.cmake\n"
    "#include <cstddef>\n"
    "extern const unsigned char ${NAME}[] = { ${bytes} };\n"
    "extern const std::size_t ${NAME}_size = ${size};\n")
