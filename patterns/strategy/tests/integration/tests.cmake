add_test(NAME smoke COMMAND strategy)
set_tests_properties(smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "Hello, world!\nPicture was scaled!\n")

add_test(NAME random COMMAND strategy 12)
set_tests_properties(random PROPERTIES PASS_REGULAR_EXPRESSION "^Hello")