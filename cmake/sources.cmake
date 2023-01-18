# Functionality to collect headers, sources

function(collect_sources SOURCES PATHS)
  set(HEADERS_LOCAL)
  set(SOURCES_LOCAL)
  
  foreach(PATH_ENTRY ${PATHS})
    file(GLOB_RECURSE SOURCES_TO_FILTER "${PATH_ENTRY}/*.c" "${PATH_ENTRY}/*.cc" "${PATH_ENTRY}/*.cpp")
    list(APPEND SOURCES_LOCAL ${SOURCES_TO_FILTER})
    set(SOURCES_TO_FILTER)
  endforeach()

  set(${SOURCES} ${SOURCES_LOCAL} PARENT_SCOPE)
endfunction()

function(create_test NAME SOURCES LIBS)
    add_executable("${NAME}" ${CMAKE_SOURCE_DIR}/main.cc ${SOURCES})
    target_link_libraries("${NAME}" ${LIBS})
    add_test(NAME ${NAME} COMMAND ${NAME} --gtest-output=xml:${CMAKE_BINARY_DIR}/test_results/)
endfunction()
    
message(STATUS "Hello")