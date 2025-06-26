#include <cstdio>

int main() {
  char str[] = "hello";  // write-access
  // char* str = "hello";	// read-only
  char* p_str = &str[0];
  *p_str = 'b';
  *(p_str + 2) = 'v';
  printf("%s\n", str);
  *(p_str + 4) = 'a';
  for (char* p = p_str; *p != '\0'; ++p) {
    printf("%c", *p);
  }
  return 0;
}