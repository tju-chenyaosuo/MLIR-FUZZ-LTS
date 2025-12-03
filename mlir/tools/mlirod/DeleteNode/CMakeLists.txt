# For a better template to copy, see examples/standalone
include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
)

add_mlir_tool(DeleteNode
  main.cpp
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(DeleteNode
  PRIVATE
  ${dialect_libs}
  MLIRMlirOptMain
  MLIRAnalysis
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTransforms)
